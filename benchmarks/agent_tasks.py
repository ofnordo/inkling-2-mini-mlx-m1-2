"""Agentic benchmark tasks for low-knowledge LLMs.

Every task is set in a *fictional* world: all facts the model needs are given in
context (system prompt, files in a sandbox, or tool responses), so the suite
measures agentic skill (tool use, planning, reading, debugging, recovery)
instead of memorized world knowledge.

Each task exposes:
  - name / category / system / user
  - tools(): JSON-schema tool declarations (OpenAI-style)
  - setup(sandbox: Path): create files / env state
  - call(name, args) -> str: execute a tool
  - done: bool (set when the model calls finish/submit)
  - check(final_text) -> dict(success, score, detail)
"""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import json
import math
import random
import re
import subprocess
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any


def _tool(name: str, desc: str, props: dict[str, Any], required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required if required is not None else list(props.keys()),
            },
        },
    }


S = {"type": "string"}
I = {"type": "integer"}


class ToolError(Exception):
    pass


class Task:
    name = "base"
    category = "base"
    system = ""
    user = ""
    max_turns = 24

    def __init__(self) -> None:
        self.done = False
        self.submitted: Any = None
        self.sandbox: Path | None = None
        self.log: list[tuple[str, dict]] = []

    def tools(self) -> list[dict]:
        return []

    def setup(self, sandbox: Path) -> None:
        self.sandbox = sandbox.resolve()

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if fn is None or name not in {t["function"]["name"] for t in self.tools()}:
            raise ToolError(f"unknown tool '{name}'")
        self.log.append((name, dict(args)))
        try:
            return fn(**args)
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}") from e

    def check(self, final_text: str) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sandboxed file / exec tools shared by repo + data tasks
# ---------------------------------------------------------------------------


class FileTools(Task):
    run_timeout = 60

    def _p(self, path: str) -> Path:
        assert self.sandbox is not None
        p = (self.sandbox / path).resolve()
        if not str(p).startswith(str(self.sandbox.resolve())):
            raise ToolError("path escapes the workspace")
        return p

    def file_tools(self) -> list[dict]:
        return [
            _tool("list_dir", "List files under a directory of the workspace (recursive).", {"path": S}),
            _tool("read_file", "Read a text file from the workspace. Returns numbered lines.", {"path": S}),
            _tool("write_file", "Create or overwrite a file in the workspace with the given content.", {"path": S, "content": S}),
            _tool(
                "replace_in_file",
                "Replace an exact substring `old` with `new` in a file. `old` must occur exactly once.",
                {"path": S, "old": S, "new": S},
            ),
            _tool("search", "Search all workspace files for a case-insensitive substring. Returns path:line: text.", {"query": S}),
        ]

    def t_list_dir(self, path: str = ".") -> str:
        root = self._p(path)
        if not root.exists():
            raise ToolError(f"no such directory: {path}")
        out = []
        for f in sorted(root.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and not f.name.startswith("."):
                out.append(str(f.relative_to(self.sandbox)))
        return "\n".join(out) or "(empty)"

    def t_read_file(self, path: str) -> str:
        p = self._p(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        lines = p.read_text().splitlines()
        body = "\n".join(f"{i+1:4d}| {l}" for i, l in enumerate(lines[:400]))
        if len(lines) > 400:
            body += f"\n... ({len(lines) - 400} more lines truncated)"
        return body

    def t_write_file(self, path: str, content: str) -> str:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    def t_replace_in_file(self, path: str, old: str, new: str) -> str:
        p = self._p(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        txt = p.read_text()
        n = txt.count(old)
        if n != 1:
            raise ToolError(f"`old` occurs {n} times in {path}; it must occur exactly once")
        p.write_text(txt.replace(old, new))
        return f"replaced 1 occurrence in {path}"

    def t_search(self, query: str) -> str:
        assert self.sandbox is not None
        q = query.lower()
        hits = []
        for f in sorted(self.sandbox.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix in (".csv",):
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if q in line.lower():
                        hits.append(f"{f.relative_to(self.sandbox)}:{i}: {line.strip()[:200]}")
            except UnicodeDecodeError:
                continue
        if not hits:
            return "no matches"
        more = f"\n... ({len(hits) - 60} more)" if len(hits) > 60 else ""
        return "\n".join(hits[:60]) + more

    def _run(self, cmd: list[str], cwd: Path | None = None) -> str:
        try:
            r = subprocess.run(
                cmd,
                cwd=cwd or self.sandbox,
                capture_output=True,
                text=True,
                timeout=self.run_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {self.run_timeout}s"
        out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
        if len(out) > 6000:
            out = out[:2500] + "\n...\n" + out[-3000:]
        return f"exit_code={r.returncode}\n{out}"


# ===========================================================================
# 1. Customer-support API task (policy following + chained tool calls)
# ===========================================================================


class SupportRefund(Task):
    name = "support_refund"
    category = "API tool use + policy"
    max_turns = 16

    system = textwrap.dedent(
        """\
        You are a support agent for Krell Home Goods. You act ONLY through the provided tools.
        Today's date is 2026-03-20.

        ## Krell Resolution Policy (authoritative)
        1. Always identify the customer by email with get_customer_by_email before acting.
        2. DAMAGED ITEM: refund the full item price with refund_order(order_id, amount_cents, reason_code="DMG").
           Only orders with status "delivered" AND delivered_on within the last 30 days (on or after 2026-02-18) are eligible.
        3. LATE DELIVERY (delivered_on later than promised_on): do NOT refund. Instead issue store credit with
           issue_credit(customer_id, order_id, amount_cents, reason_code="LATE").
           Credit = 15% of the item price for standard-tier customers, 20% for gold-tier customers.
           Compute in integer cents and round DOWN (floor).
        4. Never refund or credit an order that is not "delivered".
        5. After all refunds/credits, send exactly ONE message with send_message(customer_id, template_id="TPL-RESOLUTION",
           vars={"refund_total_cents": <int>, "credit_total_cents": <int>}).
        6. Then call finish(summary) with a one-sentence summary.
        All money amounts are integers in cents.
        """
    )
    user = (
        "Hi, this is Jo Marsh (jo.marsh@example.com). My blue kettle arrived with a cracked lid, "
        "and my desk lamp showed up late. Can you sort it out?"
    )

    customers = {
        "C-1042": {"customer_id": "C-1042", "name": "Jo Marsh", "email": "jo.marsh@example.com", "tier": "gold"},
        "C-2210": {"customer_id": "C-2210", "name": "Jo Marsh", "email": "jo.marsh@example.org", "tier": "standard"},
    }
    orders = {
        "O-88121": {"order_id": "O-88121", "customer_id": "C-1042", "item": "Blue enamel kettle", "price_cents": 4599,
                    "status": "delivered", "promised_on": "2026-03-15", "delivered_on": "2026-03-14"},
        "O-88122": {"order_id": "O-88122", "customer_id": "C-1042", "item": "Brass desk lamp", "price_cents": 8900,
                    "status": "delivered", "promised_on": "2026-03-01", "delivered_on": "2026-03-06"},
        "O-87011": {"order_id": "O-87011", "customer_id": "C-1042", "item": "Blue ceramic mug set", "price_cents": 2400,
                    "status": "delivered", "promised_on": "2026-01-12", "delivered_on": "2026-01-10"},
        "O-88130": {"order_id": "O-88130", "customer_id": "C-1042", "item": "Kettle descaler", "price_cents": 899,
                    "status": "shipped", "promised_on": "2026-03-22", "delivered_on": None},
        "O-77001": {"order_id": "O-77001", "customer_id": "C-2210", "item": "Blue enamel kettle", "price_cents": 4599,
                    "status": "delivered", "promised_on": "2026-03-10", "delivered_on": "2026-03-12"},
    }

    def __init__(self) -> None:
        super().__init__()
        self.refunds: list[dict] = []
        self.credits: list[dict] = []
        self.messages: list[dict] = []
        self.looked_up = False

    def tools(self) -> list[dict]:
        return [
            _tool("get_customer_by_email", "Look up a customer record by exact email.", {"email": S}),
            _tool("list_orders", "List all orders (summary) for a customer id.", {"customer_id": S}),
            _tool("get_order", "Get full details for one order.", {"order_id": S}),
            _tool("refund_order", "Refund an order.", {"order_id": S, "amount_cents": I, "reason_code": S}),
            _tool("issue_credit", "Issue store credit tied to an order.",
                  {"customer_id": S, "order_id": S, "amount_cents": I, "reason_code": S}),
            _tool("send_message", "Send a templated message to a customer.",
                  {"customer_id": S, "template_id": S, "vars": {"type": "object"}}),
            _tool("finish", "End the conversation with a one-sentence summary.", {"summary": S}),
        ]

    def t_get_customer_by_email(self, email: str) -> str:
        for c in self.customers.values():
            if c["email"].lower() == email.strip().lower():
                self.looked_up = True
                return json.dumps(c)
        raise ToolError(f"no customer with email {email}")

    def t_list_orders(self, customer_id: str) -> str:
        rows = [{"order_id": o["order_id"], "item": o["item"], "status": o["status"]}
                for o in self.orders.values() if o["customer_id"] == customer_id]
        return json.dumps(rows)

    def t_get_order(self, order_id: str) -> str:
        if order_id not in self.orders:
            raise ToolError(f"no order {order_id}")
        return json.dumps(self.orders[order_id])

    def t_refund_order(self, order_id: str, amount_cents: int, reason_code: str) -> str:
        if order_id not in self.orders:
            raise ToolError(f"no order {order_id}")
        self.refunds.append({"order_id": order_id, "amount_cents": int(amount_cents), "reason_code": reason_code})
        return json.dumps({"ok": True, "refund_id": f"R-{len(self.refunds):04d}"})

    def t_issue_credit(self, customer_id: str, order_id: str, amount_cents: int, reason_code: str) -> str:
        self.credits.append({"customer_id": customer_id, "order_id": order_id,
                             "amount_cents": int(amount_cents), "reason_code": reason_code})
        return json.dumps({"ok": True, "credit_id": f"CR-{len(self.credits):04d}"})

    def t_send_message(self, customer_id: str, template_id: str, vars: dict) -> str:
        self.messages.append({"customer_id": customer_id, "template_id": template_id, "vars": vars})
        return json.dumps({"ok": True})

    def t_finish(self, summary: str) -> str:
        self.done = True
        return "conversation closed"

    def check(self, final_text: str) -> dict:
        exp_refund = [{"order_id": "O-88121", "amount_cents": 4599, "reason_code": "DMG"}]
        exp_credit = [{"customer_id": "C-1042", "order_id": "O-88122", "amount_cents": 1780, "reason_code": "LATE"}]
        exp_vars = {"refund_total_cents": 4599, "credit_total_cents": 1780}
        parts = {
            "identified_customer": self.looked_up,
            "correct_refund": self.refunds == exp_refund,
            "correct_credit": self.credits == exp_credit,
            "one_correct_message": len(self.messages) == 1
            and self.messages[0]["customer_id"] == "C-1042"
            and self.messages[0]["template_id"] == "TPL-RESOLUTION"
            and {k: int(v) for k, v in (self.messages[0]["vars"] or {}).items() if str(v).lstrip("-").isdigit()} == exp_vars,
            "finished": self.done,
        }
        score = sum(parts.values()) / len(parts)
        return {"success": all(parts.values()), "score": score,
                "detail": json.dumps({**parts, "refunds": self.refunds, "credits": self.credits, "messages": self.messages})}


# ===========================================================================
# 2. Ops / deployment task (runbook following + error recovery)
# ===========================================================================


class OpsDeploy(Task):
    name = "ops_deploy"
    category = "Stateful ops + error recovery"
    max_turns = 24

    system = textwrap.dedent(
        """\
        You are an SRE agent at Vantor Labs operating the internal deploy system through tools only.

        ## Deploy Runbook (authoritative)
        - Environments: "staging" and "prod". Every release goes to staging first, and staging must report
          health "healthy" (get_health) before the same version may be deployed to prod.
        - Each service has its own database: <service>-staging-db and <service>-prod-db.
        - Database migrations required by a release must be applied with run_migration(db, migration_id)
          to an environment's database BEFORE deploying that release to that environment.
        - Any action on prod (deploy to prod, migration on a prod db) requires an approval ticket:
          call request_approval(service, version) and pass the returned ticket id as approval_ticket.
        - Only touch the service you were asked to ship. Do not roll back or redeploy other services.
        - The deploy controller occasionally returns transient errors (HTTP 503); simply retry those.
        - When the release is live and healthy in prod, call finish(summary).

        ## Release notes: billing v2.3
        - New invoice line storage. Requires DB migration M-0057 (creates table invoice_lines_v3).
        - No config changes.
        """
    )
    user = "Please ship billing v2.3 to production."

    def __init__(self) -> None:
        super().__init__()
        self.versions = {("billing", "staging"): "2.2", ("billing", "prod"): "2.2",
                         ("search", "staging"): "5.1", ("search", "prod"): "5.1"}
        self.migrations: dict[str, list[str]] = defaultdict(list)
        self.tickets: dict[str, tuple[str, str]] = {}
        self.events: list[str] = []
        self.violations: list[str] = []
        self.deploy_calls = 0
        self.transient_fired = False

    def tools(self) -> list[dict]:
        return [
            _tool("get_status", "Show deployed versions of all services in all environments.", {}, []),
            _tool("get_health", "Health of a service in an environment.", {"service": S, "env": S}),
            _tool("get_logs", "Recent logs of a service in an environment.", {"service": S, "env": S}),
            _tool("list_migrations", "List migrations applied to a database.", {"db": S}),
            _tool("request_approval", "Request a change approval ticket for a prod change.", {"service": S, "version": S}),
            _tool("run_migration", "Apply a migration to a database. approval_ticket required for prod dbs (use \"\" otherwise).",
                  {"db": S, "migration_id": S, "approval_ticket": S}, ["db", "migration_id"]),
            _tool("deploy", "Deploy a version of a service to an environment. approval_ticket required for prod (use \"\" otherwise).",
                  {"service": S, "env": S, "version": S, "approval_ticket": S}, ["service", "env", "version"]),
            _tool("rollback", "Roll a service back to its previous version in an environment.", {"service": S, "env": S}),
            _tool("finish", "Report completion.", {"summary": S}),
        ]

    def _healthy(self, service: str, env: str) -> bool:
        v = self.versions.get((service, env))
        if service == "billing" and v == "2.3":
            return "M-0057" in self.migrations[f"billing-{env}-db"]
        return True

    def t_get_status(self) -> str:
        return json.dumps({f"{s}/{e}": v for (s, e), v in self.versions.items()})

    def t_get_health(self, service: str, env: str) -> str:
        if (service, env) not in self.versions:
            raise ToolError("unknown service/env")
        return json.dumps({"service": service, "env": env, "version": self.versions[(service, env)],
                           "health": "healthy" if self._healthy(service, env) else "degraded"})

    def t_get_logs(self, service: str, env: str) -> str:
        if (service, env) not in self.versions:
            raise ToolError("unknown service/env")
        if not self._healthy(service, env):
            return ("ERROR psycopg.errors.UndefinedTable: relation \"invoice_lines_v3\" does not exist\n"
                    "ERROR request POST /invoices failed (500)\n") * 3
        return "INFO serving requests normally"

    def t_list_migrations(self, db: str) -> str:
        return json.dumps(self.migrations[db] or ["M-0001..M-0056 (baseline)"])

    def t_request_approval(self, service: str, version: str) -> str:
        tid = f"CHG-{4100 + len(self.tickets)}"
        self.tickets[tid] = (service, version)
        self.events.append(f"approval:{tid}")
        return json.dumps({"ticket": tid, "status": "approved"})

    def t_run_migration(self, db: str, migration_id: str, approval_ticket: str = "") -> str:
        if db not in ("billing-staging-db", "billing-prod-db", "search-staging-db", "search-prod-db"):
            raise ToolError(f"unknown db {db}")
        if db.startswith("search"):
            self.violations.append(f"touched {db}")
        if "prod" in db and approval_ticket not in self.tickets:
            raise ToolError("prod migration requires a valid approval_ticket")
        if migration_id != "M-0057":
            raise ToolError(f"unknown migration {migration_id}")
        if migration_id in self.migrations[db]:
            return "already applied"
        self.migrations[db].append(migration_id)
        self.events.append(f"migrate:{db}")
        return f"applied {migration_id} to {db}"

    def t_deploy(self, service: str, env: str, version: str, approval_ticket: str = "") -> str:
        if (service, env) not in self.versions:
            raise ToolError("unknown service/env")
        if service != "billing":
            self.violations.append(f"deployed {service}/{env}")
        self.deploy_calls += 1
        if not self.transient_fired:
            self.transient_fired = True
            raise ToolError("HTTP 503 deploy controller busy, try again")
        if env == "prod":
            if approval_ticket not in self.tickets:
                raise ToolError("prod deploy requires a valid approval_ticket")
            if not (self.versions[(service, "staging")] == version and self._healthy(service, "staging")):
                self.violations.append("prod deploy before healthy staging")
        self.versions[(service, env)] = version
        self.events.append(f"deploy:{service}/{env}/{version}")
        h = "healthy" if self._healthy(service, env) else "degraded"
        return json.dumps({"ok": True, "service": service, "env": env, "version": version, "post_deploy_health": h})

    def t_rollback(self, service: str, env: str) -> str:
        if service != "billing":
            self.violations.append(f"rolled back {service}/{env}")
        self.versions[(service, env)] = "2.2" if service == "billing" else "5.0"
        self.events.append(f"rollback:{service}/{env}")
        return "rolled back"

    def t_finish(self, summary: str) -> str:
        self.done = True
        return "ok"

    def check(self, final_text: str) -> dict:
        parts = {
            "prod_on_2.3": self.versions[("billing", "prod")] == "2.3",
            "prod_healthy": self._healthy("billing", "prod"),
            "staging_healthy_2.3": self.versions[("billing", "staging")] == "2.3" and self._healthy("billing", "staging"),
            "no_violations": not self.violations,
            "finished": self.done,
        }
        # migrated-before-deploy ordering on prod (staging failure is recoverable)
        ev = self.events
        try:
            parts["prod_migration_before_prod_deploy"] = ev.index("migrate:billing-prod-db") < ev.index("deploy:billing/prod/2.3")
        except ValueError:
            parts["prod_migration_before_prod_deploy"] = False
        score = sum(parts.values()) / len(parts)
        return {"success": all(parts.values()), "score": score,
                "detail": json.dumps({**parts, "events": ev, "violations": self.violations})}


# ===========================================================================
# 3. Repo bug-fix (SWE-style)
# ===========================================================================

INVOICER_README = textwrap.dedent(
    """\
    # invoicer

    Tiny billing library for Vantor Labs seat subscriptions.

    ## Business rules (authoritative)
    1. All money uses `decimal.Decimal`. Round to cents with ROUND_HALF_UP (2.345 -> 2.35), at the line level.
    2. A service period includes BOTH its start and end date (Mar 1 - Mar 31 is 31 days; a single day is 1 day).
       A period whose end is before its start is invalid (ValueError).
    3. Proration: line net = qty * unit_price * days_in_period / days_in_month, where days_in_month is the
       number of days in the month of the period's START date. Round the line net to cents.
    4. Tax: rate by region code: NW 8.75%, SE 6%, CENTRAL 7%. Region codes are case-insensitive.
       An unknown region code is an error (ValueError).
    5. Line tax = round_cents(line_net * rate). Invoice total = sum of (line net + line tax).

    Run tests with:  python -m unittest discover -s tests -v
    """
)

INVOICER_FILES_BUGGY = {
    "invoicer/__init__.py": "",
    "invoicer/money.py": textwrap.dedent(
        """\
        from decimal import Decimal, ROUND_HALF_EVEN

        CENT = Decimal("0.01")


        def round_cents(x: Decimal) -> Decimal:
            return x.quantize(CENT, rounding=ROUND_HALF_EVEN)
        """
    ),
    "invoicer/periods.py": textwrap.dedent(
        """\
        import calendar
        from datetime import date


        def days_in_period(start: date, end: date) -> int:
            if end < start:
                raise ValueError("period ends before it starts")
            return (end - start).days


        def days_in_month(d: date) -> int:
            return calendar.monthrange(d.year, d.month)[1]
        """
    ),
    "invoicer/tax.py": textwrap.dedent(
        """\
        from decimal import Decimal

        RATES = {
            "NW": Decimal("0.0875"),
            "SE": Decimal("0.06"),
            "CENTRAL": Decimal("0.07"),
        }


        def tax_rate(region: str) -> Decimal:
            return RATES.get(region.lower(), Decimal("0"))
        """
    ),
    "invoicer/invoice.py": textwrap.dedent(
        """\
        from dataclasses import dataclass
        from datetime import date
        from decimal import Decimal

        from invoicer.money import round_cents
        from invoicer.periods import days_in_month, days_in_period
        from invoicer.tax import tax_rate


        @dataclass
        class Line:
            description: str
            qty: int
            unit_price: Decimal
            start: date
            end: date


        def line_net(line: Line) -> Decimal:
            days = days_in_period(line.start, line.end)
            dim = days_in_month(line.end)
            return round_cents(Decimal(line.qty) * line.unit_price * Decimal(days) / Decimal(dim))


        def line_tax(line: Line, region: str) -> Decimal:
            return round_cents(line_net(line) * tax_rate(region))


        def invoice_total(lines: list[Line], region: str) -> Decimal:
            return sum((line_net(l) + line_tax(l, region) for l in lines), Decimal("0"))
        """
    ),
    "tests/test_invoicer.py": textwrap.dedent(
        """\
        import unittest
        from datetime import date
        from decimal import Decimal

        from invoicer.invoice import Line, invoice_total, line_net, line_tax
        from invoicer.money import round_cents
        from invoicer.periods import days_in_period
        from invoicer.tax import tax_rate

        FULL = Line("Seat", 3, Decimal("19.99"), date(2026, 4, 1), date(2026, 4, 30))
        PART = Line("Seat", 2, Decimal("30.00"), date(2026, 3, 10), date(2026, 3, 24))
        CROSS = Line("Pro", 1, Decimal("62.00"), date(2026, 1, 20), date(2026, 2, 5))


        class TestInvoicer(unittest.TestCase):
            def test_round_half_up(self):
                self.assertEqual(round_cents(Decimal("2.345")), Decimal("2.35"))
                self.assertEqual(round_cents(Decimal("0.125")), Decimal("0.13"))

            def test_days_inclusive(self):
                self.assertEqual(days_in_period(date(2026, 3, 1), date(2026, 3, 31)), 31)
                self.assertEqual(days_in_period(date(2026, 3, 5), date(2026, 3, 5)), 1)

            def test_days_reverse_raises(self):
                with self.assertRaises(ValueError):
                    days_in_period(date(2026, 3, 5), date(2026, 3, 4))

            def test_tax_case_insensitive(self):
                self.assertEqual(tax_rate("nw"), Decimal("0.0875"))
                self.assertEqual(tax_rate("Se"), Decimal("0.06"))

            def test_tax_unknown_raises(self):
                with self.assertRaises(ValueError):
                    tax_rate("ZZ")

            def test_full_month_line(self):
                self.assertEqual(line_net(FULL), Decimal("59.97"))

            def test_prorated_line(self):
                self.assertEqual(line_net(PART), Decimal("29.03"))

            def test_cross_month_line(self):
                self.assertEqual(line_net(CROSS), Decimal("34.00"))

            def test_line_tax(self):
                self.assertEqual(line_tax(FULL, "NW"), Decimal("5.25"))

            def test_invoice_total(self):
                self.assertEqual(invoice_total([FULL, PART], "central"), Decimal("95.23"))


        if __name__ == "__main__":
            unittest.main()
        """
    ),
}

INVOICER_FIXES = [
    ("invoicer/money.py", "ROUND_HALF_EVEN", "ROUND_HALF_UP"),
    ("invoicer/periods.py", "return (end - start).days", "return (end - start).days + 1"),
    ("invoicer/tax.py", 'return RATES.get(region.lower(), Decimal("0"))',
     'key = region.upper()\n    if key not in RATES:\n        raise ValueError(region)\n    return RATES[key]'),
    ("invoicer/invoice.py", "days_in_month(line.end)", "days_in_month(line.start)"),
]


class RepoBugfix(FileTools):
    name = "repo_bugfix"
    category = "SWE: debug + fix repo"
    max_turns = 30

    system = textwrap.dedent(
        """\
        You are a software engineering agent working in a small Python repository (the workspace).
        Use the tools to explore files, run the test suite, and edit code. The README contains the
        authoritative business rules. Fix the library code (do NOT edit the tests) until every test passes,
        then call finish(summary).
        """
    )
    user = "The invoicer test suite is failing. Find and fix the bugs so all tests pass."

    def tools(self) -> list[dict]:
        return self.file_tools() + [
            _tool("run_tests", "Run the unit test suite and return its output.", {}, []),
            _tool("finish", "Declare the task complete.", {"summary": S}),
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        (sandbox / "README.md").write_text(INVOICER_README)
        for rel, txt in INVOICER_FILES_BUGGY.items():
            p = sandbox / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)
        self._test_src = INVOICER_FILES_BUGGY["tests/test_invoicer.py"]

    def t_run_tests(self) -> str:
        return self._run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    def t_finish(self, summary: str) -> str:
        self.done = True
        return "ok"

    def check(self, final_text: str) -> dict:
        assert self.sandbox is not None
        tests_untouched = (self.sandbox / "tests/test_invoicer.py").read_text() == self._test_src
        # Run pristine tests regardless of edits.
        (self.sandbox / "tests/test_invoicer.py").write_text(self._test_src)
        out = self._run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        ok = len(re.findall(r"\.\.\. ok", out))
        total = 10
        score = ok / total
        return {"success": ok == total and tests_untouched, "score": score,
                "detail": f"passed {ok}/{total}; tests_untouched={tests_untouched}"}


# ===========================================================================
# 4. Implement a fictional spec (feature work, hidden tests)
# ===========================================================================

RUNECONF_SPEC = textwrap.dedent(
    """\
    # RuneConf v1 specification

    RuneConf is a small line-based configuration format. Implement `parse(text: str) -> dict` and
    `class RuneConfError(Exception)` in `runeconf.py`. RuneConfError must take `(message, line)` and expose
    the 1-based line number as the attribute `.line`.

    ## Lines
    - Blank lines and lines whose first non-space character is `#` are ignored.
    - An inline comment starts at ` #` (a space followed by `#`) that is NOT inside a double-quoted string.
    - Leading/trailing whitespace on a line is ignored.

    ## Sections
    - `[name]` or `[a.b.c]` starts a section. Name parts match `[A-Za-z0-9_-]+`, separated by dots.
    - Sections create nested dicts: `[a.b]` puts subsequent keys in `result["a"]["b"]`.
    - Keys before any section header go into the root dict.
    - A section header may appear more than once; keys are merged into the same dict.

    ## Assignments
    - `key = value`, whitespace around `=` optional. Keys match `[A-Za-z0-9_-]+`.
    - Assigning the same key twice in the same section (even across re-opened headers) is an error.
    - Any other non-ignored line that is not a header or a valid assignment is an error ("malformed line").

    ## Values (checked in this order)
    1. Double-quoted string: `"..."`, escapes `\\"`, `\\\\`, `\\n`, `\\t`. Unterminated string is an error.
    2. List: `[v1, v2, ...]` of scalar values (strings, ints, floats, booleans, bare words; no nested lists).
       Trailing comma allowed. `[]` is an empty list. Items are typed with the same scalar rules.
    3. Boolean (case-insensitive): `true yes on` -> True, `false no off` -> False.
    4. Integer: optional `+`/`-`, digits, single underscores allowed between digits (`1_000` -> 1000).
    5. Float: optional sign, digits `.` digits (`3.25`, `-0.5`).
    6. Otherwise: a bare string (the trimmed text, e.g. `hello world`).

    ## Interpolation
    - Only inside double-quoted strings: `${ref}` is replaced by the value of a previously defined key.
    - `ref` without dots is a key in the CURRENT section. `ref` with dots is an absolute path from the root:
      all parts but the last are section names, the last is the key (`${server.port}`).
    - Inserted text: ints/floats via `str()`, booleans as `true`/`false`, strings as-is.
    - Reference to an undefined key is an error.

    ## Errors
    All errors raise RuneConfError with `.line` set to the 1-based line number where the error occurs.
    """
)

RUNECONF_STUB = textwrap.dedent(
    """\
    \"\"\"RuneConf v1 parser. See SPEC.md.\"\"\"


    class RuneConfError(Exception):
        def __init__(self, message: str, line: int):
            super().__init__(f"line {line}: {message}")
            self.line = line


    def parse(text: str) -> dict:
        raise NotImplementedError
    """
)

RUNECONF_VISIBLE_TESTS = textwrap.dedent(
    '''\
    import unittest

    from runeconf import RuneConfError, parse


    class TestVisible(unittest.TestCase):
        def test_root_and_section(self):
            self.assertEqual(parse("name = demo\\n[server]\\nport = 8080\\n"),
                             {"name": "demo", "server": {"port": 8080}})

        def test_quoted_and_bool(self):
            self.assertEqual(parse('[a]\\ntitle = "Hi there"\\ndebug = yes\\n'),
                             {"a": {"title": "Hi there", "debug": True}})

        def test_list(self):
            self.assertEqual(parse("xs = [1, two, 3.5]\\n"), {"xs": [1, "two", 3.5]})

        def test_duplicate_key(self):
            with self.assertRaises(RuneConfError) as cm:
                parse("a = 1\\nb = 2\\na = 3\\n")
            self.assertEqual(cm.exception.line, 3)


    if __name__ == "__main__":
        unittest.main()
    '''
)

RUNECONF_HIDDEN_TESTS = textwrap.dedent(
    '''\
    import unittest

    from runeconf import RuneConfError, parse


    class TestHidden(unittest.TestCase):
        def test_comments_and_blank(self):
            src = "# top\\n\\n   # indented comment\\nk = v  # trailing\\n"
            self.assertEqual(parse(src), {"k": "v"})

        def test_hash_inside_quotes(self):
            self.assertEqual(parse('c = "#000 and # ok"  # real comment\\n'), {"c": "#000 and # ok"})

        def test_nested_sections(self):
            src = "[a.b]\\nx = 1\\n[a.c]\\ny = 2\\n[a]\\nz = 3\\n"
            self.assertEqual(parse(src), {"a": {"b": {"x": 1}, "c": {"y": 2}, "z": 3}})

        def test_reopened_section_merge_and_dup(self):
            self.assertEqual(parse("[s]\\na = 1\\n[t]\\nb = 2\\n[s]\\nc = 3\\n"),
                             {"s": {"a": 1, "c": 3}, "t": {"b": 2}})
            with self.assertRaises(RuneConfError) as cm:
                parse("[s]\\na = 1\\n[t]\\nb = 2\\n[s]\\na = 9\\n")
            self.assertEqual(cm.exception.line, 6)

        def test_integers(self):
            self.assertEqual(parse("a = 1_000\\nb = -42\\nc = +7\\n"), {"a": 1000, "b": -42, "c": 7})

        def test_floats(self):
            r = parse("a = 3.25\\nb = -0.5\\n")
            self.assertEqual(r, {"a": 3.25, "b": -0.5})
            self.assertIsInstance(r["a"], float)

        def test_booleans(self):
            self.assertEqual(parse("a = ON\\nb = off\\nc = True\\nd = no\\n"),
                             {"a": True, "b": False, "c": True, "d": False})

        def test_list_edge_cases(self):
            self.assertEqual(parse('a = []\\nb = [x, "y z", no, 2,]\\n'),
                             {"a": [], "b": ["x", "y z", False, 2]})

        def test_escapes(self):
            self.assertEqual(parse(r'a = "q\\"uote\\\\ tab\\tnl\\n"' + "\\n"), {"a": 'q"uote\\\\ tab\\tnl\\n'})

        def test_bare_string_with_spaces(self):
            self.assertEqual(parse("greeting = hello world\\n"), {"greeting": "hello world"})

        def test_interpolation_local_and_absolute(self):
            src = ('[server]\\nhost = example.internal\\nport = 8443\\ntls = yes\\n'
                   'url = "https://${host}:${port}"\\n[client]\\ntarget = "${server.url}/api"\\n'
                   'secure = "${server.tls}"\\n')
            r = parse(src)
            self.assertEqual(r["server"]["url"], "https://example.internal:8443")
            self.assertEqual(r["client"]["target"], "https://example.internal:8443/api")
            self.assertEqual(r["client"]["secure"], "true")

        def test_undefined_reference(self):
            with self.assertRaises(RuneConfError) as cm:
                parse('[a]\\nx = 1\\ny = "${missing}"\\n')
            self.assertEqual(cm.exception.line, 3)

        def test_malformed_line(self):
            with self.assertRaises(RuneConfError) as cm:
                parse("a = 1\\nthis line has no equals\\n")
            self.assertEqual(cm.exception.line, 2)

        def test_unterminated_string(self):
            with self.assertRaises(RuneConfError) as cm:
                parse('a = 1\\n\\nb = "oops\\n')
            self.assertEqual(cm.exception.line, 3)


    if __name__ == "__main__":
        unittest.main()
    '''
)

RUNECONF_REFERENCE = textwrap.dedent(
    r'''
    import re


    class RuneConfError(Exception):
        def __init__(self, message, line):
            super().__init__(f"line {line}: {message}")
            self.line = line


    NAME = r"[A-Za-z0-9_-]+"
    HDR = re.compile(rf"^\[({NAME}(?:\.{NAME})*)\]$")
    ASSIGN = re.compile(rf"^({NAME})\s*=\s*(.*)$")


    def _strip_comment(s):
        inq = False
        i = 0
        while i < len(s):
            c = s[i]
            if inq and c == "\\":
                i += 2
                continue
            if c == '"':
                inq = not inq
            elif c == "#" and not inq and i > 0 and s[i - 1] == " ":
                return s[: i - 1]
            i += 1
        return s


    def _scalar(v):
        low = v.lower()
        if low in ("true", "yes", "on"):
            return True
        if low in ("false", "no", "off"):
            return False
        if re.fullmatch(r"[+-]?\d+(?:_\d+)*", v):
            return int(v.replace("_", ""))
        if re.fullmatch(r"[+-]?\d+\.\d+", v):
            return float(v)
        return v


    def _fmt(x):
        if isinstance(x, bool):
            return "true" if x else "false"
        return str(x)


    def parse(text):
        root = {}
        cur = root
        path = []
        seen = set()
        for ln, raw in enumerate(text.splitlines(), 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            s = _strip_comment(s).strip()
            m = HDR.match(s)
            if m:
                path = m.group(1).split(".")
                cur = root
                for p in path:
                    cur = cur.setdefault(p, {})
                continue
            m = ASSIGN.match(s)
            if not m:
                raise RuneConfError("malformed line", ln)
            key, v = m.group(1), m.group(2).strip()
            fq = tuple(path + [key])
            if fq in seen:
                raise RuneConfError(f"duplicate key {key}", ln)
            if v.startswith('"'):
                out = []
                i = 1
                closed = False
                while i < len(v):
                    c = v[i]
                    if c == "\\" and i + 1 < len(v):
                        out.append({'"': '"', "\\": "\\", "n": "\n", "t": "\t"}.get(v[i + 1], v[i + 1]))
                        i += 2
                        continue
                    if c == '"':
                        closed = True
                        break
                    out.append(c)
                    i += 1
                if not closed:
                    raise RuneConfError("unterminated string", ln)
                sval = "".join(out)

                def rep(mm):
                    ref = mm.group(1).split(".")
                    if len(ref) == 1:
                        tgt = tuple(path + ref)
                    else:
                        tgt = tuple(ref)
                    if tgt not in seen:
                        raise RuneConfError(f"undefined reference {mm.group(1)}", ln)
                    node = root
                    for p in tgt:
                        node = node[p]
                    return _fmt(node)

                val = re.sub(r"\$\{([^}]+)\}", rep, sval)
            elif v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                items = [x.strip() for x in inner.split(",")] if inner else []
                if items and items[-1] == "":
                    items = items[:-1]
                val = [x[1:-1] if len(x) >= 2 and x[0] == x[-1] == '"' else _scalar(x) for x in items]
            else:
                val = _scalar(v)
            cur[key] = val
            seen.add(fq)
        return root
    '''
)


class SpecImplement(FileTools):
    name = "spec_implement"
    category = "SWE: implement from spec (hidden tests)"
    max_turns = 30

    system = textwrap.dedent(
        """\
        You are a software engineering agent. The workspace contains SPEC.md describing a configuration
        format called RuneConf, a stub `runeconf.py`, and a few visible tests in tests/. Implement the full
        specification in runeconf.py using only the Python standard library. The visible tests are only a
        sample: a larger hidden test suite will check every rule in SPEC.md. Run the tests, iterate, then
        call finish(summary).
        """
    )
    user = "Implement the RuneConf parser described in SPEC.md."

    def tools(self) -> list[dict]:
        return self.file_tools() + [
            _tool("run_tests", "Run the visible unit tests.", {}, []),
            _tool("run_python", "Run a short Python snippet in the workspace and return stdout/stderr.", {"code": S}),
            _tool("finish", "Declare the task complete.", {"summary": S}),
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        (sandbox / "SPEC.md").write_text(RUNECONF_SPEC)
        (sandbox / "runeconf.py").write_text(RUNECONF_STUB)
        (sandbox / "tests").mkdir(exist_ok=True)
        (sandbox / "tests/test_visible.py").write_text(RUNECONF_VISIBLE_TESTS)

    def t_run_tests(self) -> str:
        return self._run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    def t_run_python(self, code: str) -> str:
        return self._run([sys.executable, "-c", code])

    def t_finish(self, summary: str) -> str:
        self.done = True
        return "ok"

    def check(self, final_text: str) -> dict:
        assert self.sandbox is not None
        hidden = self.sandbox / ".hidden"
        hidden.mkdir(exist_ok=True)
        (hidden / "test_hidden.py").write_text(
            "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            + RUNECONF_HIDDEN_TESTS
        )
        out = self._run([sys.executable, ".hidden/test_hidden.py", "-v"])
        out2 = self._run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        ok = len(re.findall(r"\.\.\. ok", out))
        vis = len(re.findall(r"\.\.\. ok", out2))
        total = 14
        return {"success": ok == total, "score": ok / total,
                "detail": f"hidden {ok}/{total}, visible {vis}/4"}


# ===========================================================================
# 5. Multi-hop retrieval over a fictional wiki
# ===========================================================================

WIKI = {
    "README.md": """\
        # Vantor Labs internal wiki
        Directories: services/, teams/, oncall/, policy/, incidents/.
        Pages may be outdated; policy/changelog.md records policy changes and overrides older pages.
        """,
    "services/catalog.md": """\
        # Service catalog
        | Service | Tier | Owning team | Scheduled jobs |
        |---|---|---|---|
        | Tally | Tier-1 | Abacus | ledger-sync, fx-rates-pull |
        | Quill | Tier-2 | Inkwell | doc-render |
        | Tally-Legacy | deprecated | Oldguard | ledger-sync-v1 |
        | Beacon | Tier-1 | Lighthouse | ping-sweep |
        """,
    "teams/abacus.md": """\
        # Team Abacus
        Owns: Tally. Lead: Priya Natarajan.
        Members: Priya Natarajan, Tomás Ferreira, Lena Okafor, Dmitri Volkov.
        Rotation file: oncall/abacus-2026-q1.md
        """,
    "teams/oldguard.md": """\
        # Team Oldguard
        Owns legacy systems including Tally-Legacy. Members: Hugo Brandt, Sana Iqbal.
        Rotation file: oncall/oldguard.md
        """,
    "oncall/abacus-2026-q1.md": """\
        # Abacus on-call rotation, 2026 Q1
        | Week starting (Mon) | Primary | Secondary |
        |---|---|---|
        | 2026-02-23 | Priya Natarajan | Lena Okafor |
        | 2026-03-02 | Tomás Ferreira | Lena Okafor |
        | 2026-03-09 | Lena Okafor | Dmitri Volkov |
        | 2026-03-16 | Dmitri Volkov | Priya Natarajan |
        | 2026-03-23 | Priya Natarajan | Tomás Ferreira |
        | 2026-03-30 | Tomás Ferreira | Dmitri Volkov |

        ## Swaps (override the table above)
        - Week of 2026-03-16: PRIMARY covered by Lena Okafor instead of Dmitri Volkov (Dmitri on leave). Approved by Priya 2026-03-10.
        """,
    "oncall/abacus-2025.md": """\
        # ARCHIVED: Abacus rotation 2025 (do not use)
        | Week starting | Primary | Secondary |
        |---|---|---|
        | 2025-03-16 | Marco Bellini | Tomás Ferreira |
        | 2025-03-23 | Lena Okafor | Marco Bellini |
        """,
    "oncall/oldguard.md": """\
        # Oldguard rotation
        Weekly, primary only. Week of 2026-03-16: Hugo Brandt. Week of 2026-03-23: Sana Iqbal.
        """,
    "policy/oncall-stipend.md": """\
        # On-call stipend policy
        - Primary week: $400. Secondary week: $150.
        - A week counts toward the month in which the week STARTS (its Monday).
        - When shifts are swapped, the stipend goes to the person who actually covered the shift.
        """,
    "policy/deploys.md": """\
        # Production deploy policy (last edited 2025-11-04)
        - Tier-1 services: at most 4 production deploys per day.
        - Tier-2 services: at most 8 production deploys per day.
        - No deploys during company freeze windows.
        """,
    "policy/changelog.md": """\
        # Policy changelog (authoritative, newest first)
        - 2026-02-01: Tier-1 production deploy limit reduced from 4 to 3 per day (after INC-2231). Tier-2 unchanged.
        - 2025-12-15: Freeze window added for Dec 22 - Jan 2.
        - 2025-09-30: On-call secondary stipend raised from $100 to $150.
        """,
    "incidents/INC-2231.md": """\
        # INC-2231: Tally double-posted ledger entries
        Date: 2026-01-19. Cause: five Tally deploys in one day overlapped with ledger-sync.
        Follow-up: reduce Tier-1 deploy limit (see policy changelog).
        """,
}


class WikiQA(FileTools):
    category = "Multi-hop retrieval"
    max_turns = 14

    system = textwrap.dedent(
        """\
        You are a research agent for Vantor Labs. Today is Wednesday 2026-03-18.
        You know nothing about Vantor Labs beyond what is in the internal wiki (the workspace).
        Use search, list_dir and read_file to find the answer; check for overrides, swaps and newer policy.
        When confident, call submit_answer with a short final answer (a name, number or amount only).
        """
    )

    def __init__(self, name: str, question: str, answers: list[str]) -> None:
        super().__init__()
        self.name = name
        self.user = question
        self.answers = answers

    def tools(self) -> list[dict]:
        return [t for t in self.file_tools() if t["function"]["name"] in ("list_dir", "read_file", "search")] + [
            _tool("submit_answer", "Submit the final answer.", {"answer": S})
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        for rel, txt in WIKI.items():
            p = sandbox / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(txt))

    def t_submit_answer(self, answer: str) -> str:
        self.done = True
        self.submitted = answer
        return "submitted"

    def check(self, final_text: str) -> dict:
        ans = str(self.submitted if self.submitted is not None else final_text)
        norm = re.sub(r"[^a-z0-9.]", "", ans.lower().replace("ás", "as").replace(",", ""))
        ok = any(re.sub(r"[^a-z0-9.]", "", a.lower().replace("ás", "as")) == norm.rstrip(".") or
                 re.sub(r"[^a-z0-9.]", "", a.lower()) in norm for a in self.answers)
        return {"success": ok, "score": float(ok), "detail": f"submitted={ans!r} expected={self.answers}"}


# ===========================================================================
# 6. Data analysis with code execution
# ===========================================================================

DATA_DICT = textwrap.dedent(
    """\
    # sales.csv data dictionary
    Columns: order_id, date (YYYY-MM-DD), region, sku, units, unit_price_cents, status

    - unit_price_cents is the price per unit in integer CENTS.
    - status is one of: completed, returned, cancelled.
      * completed rows ADD units * unit_price_cents to revenue.
      * returned rows SUBTRACT units * unit_price_cents from revenue (a return is booked as its own row).
      * cancelled rows are ignored entirely.
    - "Net revenue" = completed amount minus returned amount.
    - Vantor's FISCAL year starts on February 1:
      FQ1 = Feb-Apr, FQ2 = May-Jul, FQ3 = Aug-Oct, FQ4 = Nov-Jan.
    - Return rate of a SKU = returned units / completed units.
    """
)


def _make_sales() -> list[dict]:
    rng = random.Random(7)
    regions = ["North", "South", "East", "West"]
    skus = {"VX-100": 1299, "VX-200": 2499, "VX-310": 899, "LM-10": 4999, "LM-20": 7499,
            "QK-1": 350, "QK-2": 575, "ZP-9": 15999}
    region_w = {"North": 1.0, "South": 1.15, "East": 0.95, "West": 1.05}
    ret_bias = {"VX-310": 0.17, "QK-2": 0.12}
    rows = []
    oid = 50000
    for day in range(214):  # 2025-06-01 .. 2025-12-31
        y, m, d = 2025, 6, 1
        # compute date
        import datetime as _dt
        dt = _dt.date(2025, 6, 1) + _dt.timedelta(days=day)
        for _ in range(rng.randint(2, 5)):
            region = rng.choices(regions, weights=[region_w[r] for r in regions])[0]
            sku = rng.choice(list(skus))
            units = rng.randint(1, 6)
            p = rng.random()
            rr = ret_bias.get(sku, 0.07)
            status = "returned" if p < rr else ("cancelled" if p < rr + 0.05 else "completed")
            oid += 1
            rows.append({"order_id": oid, "date": dt.isoformat(), "region": region, "sku": sku,
                         "units": units, "unit_price_cents": skus[sku], "status": status})
    return rows


def _sales_answers(rows: list[dict]) -> dict:
    net = defaultdict(int)
    for r in rows:
        if r["date"][5:7] in ("08", "09", "10"):
            amt = r["units"] * r["unit_price_cents"]
            if r["status"] == "completed":
                net[r["region"]] += amt
            elif r["status"] == "returned":
                net[r["region"]] -= amt
    ranked = sorted(net.items(), key=lambda kv: -kv[1])
    comp = defaultdict(int)
    ret = defaultdict(int)
    for r in rows:
        if r["status"] == "completed":
            comp[r["sku"]] += r["units"]
        elif r["status"] == "returned":
            ret[r["sku"]] += r["units"]
    rates = sorted(((ret[s] / comp[s], s) for s in comp if comp[s] >= 50), reverse=True)
    return {"fq3_top": ranked[0], "fq3_second": ranked[1], "fq3_gap": ranked[0][1] - ranked[1][1],
            "top_return": rates[0], "second_return": rates[1]}


class DataAnalysis(FileTools):
    category = "Data analysis w/ code exec"
    max_turns = 16

    system = textwrap.dedent(
        """\
        You are a data analyst agent. The workspace contains sales.csv and DATA_DICTIONARY.md, which defines
        every business term (read it first; the definitions are company-specific). Use run_python to compute
        answers exactly (standard library only: csv, collections, etc.). When done call submit_answer.
        """
    )

    def __init__(self, name: str, question: str, kind: str) -> None:
        super().__init__()
        self.name = name
        self.user = question
        self.kind = kind

    def tools(self) -> list[dict]:
        return [t for t in self.file_tools() if t["function"]["name"] in ("list_dir", "read_file")] + [
            _tool("run_python", "Run a Python script in the workspace; returns stdout/stderr.", {"code": S}),
            _tool("submit_answer", "Submit the final answer.", {"answer": S}),
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        rows = _make_sales()
        self.expected = _sales_answers(rows)
        import csv
        with open(sandbox / "sales.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (sandbox / "DATA_DICTIONARY.md").write_text(DATA_DICT)

    def t_read_file(self, path: str) -> str:
        if path.endswith(".csv"):
            p = self._p(path)
            lines = p.read_text().splitlines()
            return "\n".join(lines[:15]) + f"\n... ({len(lines) - 15} more rows; use run_python to analyze)"
        return super().t_read_file(path)

    def t_run_python(self, code: str) -> str:
        return self._run([sys.executable, "-c", code])

    def t_submit_answer(self, answer: str) -> str:
        self.done = True
        self.submitted = answer
        return "submitted"

    def check(self, final_text: str) -> dict:
        ans = str(self.submitted if self.submitted is not None else final_text)
        e = self.expected
        if self.kind == "fq3":
            region, cents = e["fq3_top"]
            gap = e["fq3_gap"]
            nums = [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", ans)]
            has_region = region.lower() in ans.lower()
            has_amt = any(abs(n - cents / 100) < 0.011 for n in nums)
            has_gap = any(abs(n - gap / 100) < 0.011 for n in nums)
            score = (has_region + has_amt + has_gap) / 3
            return {"success": has_region and has_amt and has_gap, "score": score,
                    "detail": f"submitted={ans!r} expected region={region} net=${cents/100:.2f} gap=${gap/100:.2f}"}
        rate, sku = e["top_return"]
        ok = sku.lower() in ans.lower() and e["second_return"][1].lower() not in ans.lower()
        return {"success": ok, "score": float(ok),
                "detail": f"submitted={ans!r} expected sku={sku} rate={rate:.4f}"}


# ---------------------------------------------------------------------------


def all_tasks() -> list[Task]:
    return [
        SupportRefund(),
        OpsDeploy(),
        WikiQA("wiki_oncall_2hop",
               "Who is the primary on-call engineer THIS week for the service that runs the `ledger-sync` job?",
               ["Lena Okafor"]),
        WikiQA("wiki_policy_override",
               "What is the maximum number of production deploys per day currently allowed for the service that runs `ledger-sync`?",
               ["3"]),
        WikiQA("wiki_stipend_calc",
               "How much on-call stipend (in dollars) does Lena Okafor earn for March 2026?",
               ["950", "$950", "950.00"]),
        DataAnalysis("data_fq3_revenue",
                     "Which region had the highest net revenue in fiscal Q3 of this dataset? Give the region, its FQ3 "
                     "net revenue in dollars (2 decimals), and how many dollars it beat the second-place region by.",
                     "fq3"),
        DataAnalysis("data_return_rate",
                     "Among SKUs with at least 50 completed units, which SKU has the highest return rate?",
                     "returns"),
        RepoBugfix(),
        SpecImplement(),
    ]


def selftest() -> None:
    """Validate checkers & reference solutions without a model."""
    import tempfile

    # repo bugfix: buggy fails, fixed passes
    with tempfile.TemporaryDirectory() as d:
        t = RepoBugfix()
        t.setup(Path(d))
        r0 = t.check("")
        for path, old, new in INVOICER_FIXES:
            p = Path(d) / path
            p.write_text(p.read_text().replace(old, new))
        r1 = t.check("")
        print("repo_bugfix buggy:", r0["detail"], "| fixed:", r1["detail"])
        assert not r0["success"] and r1["success"]

    with tempfile.TemporaryDirectory() as d:
        t = SpecImplement()
        t.setup(Path(d))
        r0 = t.check("")
        (Path(d) / "runeconf.py").write_text(RUNECONF_REFERENCE)
        r1 = t.check("")
        print("spec_implement stub:", r0["detail"], "| reference:", r1["detail"])
        assert r1["success"], t._run([sys.executable, ".hidden/test_hidden.py", "-v"])

    rows = _make_sales()
    e = _sales_answers(rows)
    print("data:", {k: v for k, v in e.items()})

    # support: scripted optimal trajectory
    t = SupportRefund()
    t.call("get_customer_by_email", {"email": "jo.marsh@example.com"})
    t.call("refund_order", {"order_id": "O-88121", "amount_cents": 4599, "reason_code": "DMG"})
    t.call("issue_credit", {"customer_id": "C-1042", "order_id": "O-88122", "amount_cents": 1780, "reason_code": "LATE"})
    t.call("send_message", {"customer_id": "C-1042", "template_id": "TPL-RESOLUTION",
                            "vars": {"refund_total_cents": 4599, "credit_total_cents": 1780}})
    t.call("finish", {"summary": "done"})
    assert t.check("")["success"], t.check("")

    # ops: scripted optimal trajectory (with transient retry)
    t = OpsDeploy()
    t.call("run_migration", {"db": "billing-staging-db", "migration_id": "M-0057"})
    try:
        t.call("deploy", {"service": "billing", "env": "staging", "version": "2.3"})
    except ToolError:
        pass
    t.call("deploy", {"service": "billing", "env": "staging", "version": "2.3"})
    tk = json.loads(t.call("request_approval", {"service": "billing", "version": "2.3"}))["ticket"]
    t.call("run_migration", {"db": "billing-prod-db", "migration_id": "M-0057", "approval_ticket": tk})
    t.call("deploy", {"service": "billing", "env": "prod", "version": "2.3", "approval_ticket": tk})
    t.call("finish", {"summary": "done"})
    assert t.check("")["success"], t.check("")

    # wiki checker
    w = WikiQA("x", "q", ["950", "$950"])
    w.submitted = "$950"
    assert w.check("")["success"]
    w.submitted = "Lena Okafor earns $1,100"
    assert not w.check("")["success"]
    w2 = WikiQA("y", "q", ["Lena Okafor"])
    w2.submitted = "Lena Okafor"
    assert w2.check("")["success"]
    w2.submitted = "Dmitri Volkov"
    assert not w2.check("")["success"]
    print("selftest OK")


if __name__ == "__main__":
    selftest()
