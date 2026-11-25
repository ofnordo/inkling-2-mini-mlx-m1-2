"""Tier-2 Agentic Stress Benchmarks for Inkling-2-Mini:

1. `refactor_multifile` (30+ turn multi-file refactoring):
   Migrate a 9-file double-entry accounting library (`ledgerx`) from v1 single-currency
   int-cents to v2 multi-currency `Money`, account currency enforcement, FX conversion,
   entry voiding/reversals, and v1->v2 JSON migration. Verified against 18 hidden tests.

2. `needle_debug_longctx` (long-context needle-in-a-haystack debugging):
   Diagnose and fix 3 interacting production bugs in a 6-file event pipeline (`pipeline/`)
   by correlating clues buried inside ~28k tokens of noisy production logs (`logs/*.log`)
   and config snapshots (`configs/prod_snapshot.json`). Verified against a hidden replay suite.
"""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import json
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

from agent_tasks import FileTools, I, S, ToolError, _tool


class ExtendedFileTools(FileTools):
    """FileTools with line-range reading for long logs and larger workspaces."""

    def file_tools(self) -> list[dict]:
        return [
            _tool("list_dir", "List files under a directory of the workspace (recursive).", {"path": S}),
            _tool(
                "read_file",
                "Read lines from a text file in the workspace. Optional start_line/end_line (1-based inclusive, default 1..400).",
                {"path": S, "start_line": I, "end_line": I},
                ["path"],
            ),
            _tool("write_file", "Create or overwrite a file in the workspace with the given content.", {"path": S, "content": S}),
            _tool(
                "replace_in_file",
                "Replace an exact substring `old` with `new` in a file. `old` must occur exactly once.",
                {"path": S, "old": S, "new": S},
            ),
            _tool(
                "search",
                "Search workspace files for a case-insensitive substring. Optional path prefix to scope search.",
                {"query": S, "path": S},
                ["query"],
            ),
        ]

    def t_read_file(self, path: str, start_line: int = 1, end_line: int | None = None) -> str:
        p = self._p(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        lines = p.read_text().splitlines()
        s = max(1, int(start_line))
        e = min(len(lines), int(end_line) if end_line is not None else s + 399)
        if e - s + 1 > 600:
            e = s + 599
        body = "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(s, e + 1))
        if e < len(lines):
            body += f"\n... ({len(lines) - e} more lines; total={len(lines)}, use start_line={e + 1})"
        return body

    def t_search(self, query: str, path: str = ".") -> str:
        root = self._p(path)
        q = query.lower()
        hits = []
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for f in files:
            if not f.is_file() or "__pycache__" in f.parts or f.name.startswith("."):
                continue
            try:
                for i, line in enumerate(f.read_text().splitlines(), 1):
                    if q in line.lower():
                        hits.append(f"{f.relative_to(self.sandbox)}:{i}: {line.strip()[:220]}")
            except UnicodeDecodeError:
                continue
        if not hits:
            return "no matches"
        more = f"\n... ({len(hits) - 80} more)" if len(hits) > 80 else ""
        return "\n".join(hits[:80]) + more


# ===========================================================================
# TIER-2 TASK 1: Multi-File Refactoring (`ledgerx` v1 -> v2)
# ===========================================================================

LEDGERX_MIGRATION_SPEC = textwrap.dedent(
    """\
    # LedgerX v1 -> v2 Migration Specification

    Migrate the entire `ledgerx` package across all modules to v2 Multi-Currency Accounting.
    Every module must be updated cleanly and consistently. Standard library only.

    ## 1. New Module: `ledgerx/money.py`
    - `class CurrencyMismatchError(ValueError)`: raised when mixing incompatible currencies.
    - `@dataclass(frozen=True) class Money`:
      - Fields: `amount_cents: int`, `currency: str = "USD"`
      - Validation in `__post_init__`:
        - `amount_cents` must be `int` (and NOT `bool`), else `TypeError`.
        - `currency` must be a 3-letter uppercase ASCII string (`^[A-Z]{3}$`), else `ValueError`.
      - Arithmetic:
        - `m1 + m2`, `m1 - m2`: require same `currency`, else raise `CurrencyMismatchError`.
        - `-m`: returns `Money(-m.amount_cents, m.currency)`.
        - `m == other`: standard dataclass equality (`Money(0, "USD") != Money(0, "EUR")`).
        - Ordering (`<`, `<=`, `>`, `>=`): require same `currency`, else raise `CurrencyMismatchError`.
        - `Money.zero(currency: str = "USD") -> Money`.
        - `format() -> str`: returns e.g. `"USD 12.34"` or `"EUR -5.00"` (sign before digits).

    ## 2. Models (`ledgerx/models.py`)
    - `class AlreadyVoidedError(ValueError)`: raised when voiding an entry twice.
    - `class UnbalancedEntryError(ValueError)`: raised when an entry does not balance.
    - `@dataclass(frozen=True) class Posting`:
      - Fields: `account_id: str`, `money: Money`
      - Convenience properties: `amount_cents -> int` and `currency -> str`.
    - `@dataclass class JournalEntry`:
      - Fields: `entry_id: str`, `description: str`, `postings: tuple[Posting, ...]`,
        `voided: bool = False`, `void_of: str | None = None`, `void_reason: str | None = None`.

    ## 3. Accounts (`ledgerx/accounts.py`)
    - `@dataclass(frozen=True) class Account`:
      - Fields: `account_id: str`, `name: str`, `kind: str` (`"asset"|"liability"|"equity"|"revenue"|"expense"`),
        `currency: str = "USD"`.
      - Validate `kind` in allowed set and `currency` via `Money(0, currency)`.
    - `AccountRegistry`:
      - `register(account: Account) -> None` (raises `ValueError` on duplicate `account_id`).
      - `get(account_id: str) -> Account` (raises `KeyError` if missing).
      - `all_accounts(currency: str | None = None) -> list[Account]`.

    ## 4. Ledger (`ledgerx/ledger.py`)
    - `Ledger(registry: AccountRegistry)`:
      - `post_entry(entry: JournalEntry, *, fx_rates: dict[tuple[str, str], Decimal] | None = None, base_currency: str | None = None) -> JournalEntry`:
        - Reject duplicate `entry_id` (`ValueError`) or empty `postings` (`ValueError`).
        - Every posting's `account_id` must exist in `registry` (`KeyError`), and `posting.currency == account.currency` (else `CurrencyMismatchError`).
        - **Balancing**:
          - If `fx_rates is None`: for EVERY currency present in `entry.postings`, the sum of `amount_cents` for that currency must equal `0`, else raise `UnbalancedEntryError`.
          - If `fx_rates is not None`: `base_currency` must be provided (`ValueError` if None). Convert each posting's `amount_cents` into `base_currency` using:
            - if `p.currency == base_currency`: `Decimal(p.amount_cents)`
            - else look up `fx_rates[(p.currency, base_currency)]` (or `Decimal(1) / fx_rates[(base_currency, p.currency)]` if inverse is present; else raise `CurrencyMismatchError`).
            - Convert each posting individually to integer base cents with `ROUND_HALF_UP`:
              `int((Decimal(p.amount_cents) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))`
            - The sum of converted base-currency cents across all postings must equal `0`, else raise `UnbalancedEntryError`.
      - `void_entry(entry_id: str, reason: str) -> JournalEntry`:
        - Look up original entry (`KeyError` if missing).
        - If already `voided` (or if `entry.void_of is not None`), raise `AlreadyVoidedError`.
        - Mark original `entry.voided = True`.
        - Create and append reversal `JournalEntry(entry_id=f"{entry_id}:VOID", description=f"VOID {entry_id}: {reason}", postings=tuple(Posting(p.account_id, -p.money) for p in orig.postings), voided=False, void_of=entry_id, void_reason=reason)`.
        - Note: reversal bypasses FX re-validation because it is the exact negation of an already-accepted entry.
        - Return the reversal entry.
      - `balance(account_id: str) -> Money`: returns `Money(total_cents, account.currency)`.
      - `trial_balance(currency: str | None = None) -> dict[str, Money]`: maps `account_id -> balance(account_id)` for matching accounts (all accounts if `currency is None`).
      - `entries` property or attribute: list of all posted `JournalEntry` objects in order.

    ## 5. Fees (`ledgerx/fees.py`)
    - `FeeSchedule(flat_fees: dict[str, Money] | None = None, bps: int = 0)`:
      - `compute_fee(gross: Money) -> Money`:
        - `bps_cents = int((Decimal(gross.amount_cents) * Decimal(self.bps) / Decimal(10000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))`
        - `flat_cents = self.flat_fees[gross.currency].amount_cents` if `flat_fees` has `gross.currency` else `0`.
        - Returns `Money(bps_cents + flat_cents, gross.currency)`.
      - `make_fee_postings(gross: Money, payer_account_id: str, fee_revenue_account_id: str) -> tuple[Posting, Posting]`:
        - Returns `(Posting(payer_account_id, -fee), Posting(fee_revenue_account_id, fee))`.

    ## 6. Settlement (`ledgerx/settlement.py`)
    - `SettlementBatch`:
      - `net_payouts(ledger: Ledger, account_ids: list[str]) -> dict[str, dict[str, Money]]`:
        - Groups accounts by currency: `{currency: {account_id: balance}}` ONLY for accounts in `account_ids` with `balance.amount_cents != 0`.
      - `create_settlement_entries(ledger: Ledger, merchant_account_ids: list[str], clearing_accounts: dict[str, str], batch_id: str) -> list[JournalEntry]`:
        - For each `account_id` in `merchant_account_ids`: if `bal = ledger.balance(account_id)` has `bal.amount_cents > 0`, look up `clearing_id = clearing_accounts[bal.currency]` (raising `CurrencyMismatchError` if missing), post a `JournalEntry(entry_id=f"SETTLE:{batch_id}:{account_id}", description=f"Settlement {batch_id}", postings=(Posting(account_id, -bal), Posting(clearing_id, bal)))` into `ledger`, and return the list of posted settlement entries.

    ## 7. Reports (`ledgerx/reports.py`)
    - `income_statement(ledger: Ledger, currency: str) -> dict[str, Money]`:
      - Filters accounts by `currency`.
      - `revenue`: sum of balances for `"revenue"` accounts (as `Money`).
      - `expense`: sum of balances for `"expense"` accounts (as `Money`).
      - `net_income`: `revenue + expense` (since expenses are negative or signed in postings; wait: follow the exact sign convention already in v1 `reports.py`! Check `reports.py`: in v1, revenue is positive, expenses are negative, `net_income = revenue + expense`).
    - `balance_sheet(ledger: Ledger, currency: str) -> dict[str, Money]`:
      - Filters accounts by `currency`, returns `{"asset": ..., "liability": ..., "equity": ...}` as `Money`.
    - `account_statement(ledger: Ledger, account_id: str) -> list[dict]`:
      - Returns list of `{"entry_id": ..., "description": ..., "delta": Money, "running_balance": Money, "voided": bool}` for every entry touching `account_id`.

    ## 8. Serialization (`ledgerx/serialization.py`)
    - `export_json(ledger: Ledger) -> str`:
      - JSON with `"schema_version": 2`, `"accounts": [...]` (including `"currency"`), `"entries": [...]` (each posting has `{"account_id": ..., "amount_cents": ..., "currency": ...}`; each entry includes `"voided"`, `"void_of"`, `"void_reason"`).
    - `import_json(payload: str) -> Ledger`:
      - Supports BOTH `"schema_version": 2` AND legacy v1 payloads (no `"schema_version"` or `"schema_version": 1`, where accounts and postings have no `"currency"` field -> default to `"USD"`).
      - Reconstructs `AccountRegistry` and `Ledger` with all entries and voided flags intact (note: v2 entries that were posted with FX or voided must load cleanly without re-failing FX checks!).

    ## 9. CLI Helpers (`ledgerx/cli_helpers.py`)
    - `format_ledger_summary(ledger: Ledger) -> str`:
      - Lists accounts sorted by `(currency, account_id)`, formatting each line as `f"{acc.account_id} ({acc.name}): {bal.format()}"`.
    """
)

LEDGERX_V1_FILES = {
    "ledgerx/__init__.py": "",
    "ledgerx/models.py": textwrap.dedent(
        """\
        from dataclasses import dataclass


        class UnbalancedEntryError(ValueError):
            pass


        @dataclass(frozen=True)
        class Posting:
            account_id: str
            amount_cents: int


        @dataclass
        class JournalEntry:
            entry_id: str
            description: str
            postings: tuple[Posting, ...]
        """
    ),
    "ledgerx/accounts.py": textwrap.dedent(
        """\
        from dataclasses import dataclass

        VALID_KINDS = frozenset({"asset", "liability", "equity", "revenue", "expense"})


        @dataclass(frozen=True)
        class Account:
            account_id: str
            name: str
            kind: str

            def __post_init__(self) -> None:
                if self.kind not in VALID_KINDS:
                    raise ValueError(f"invalid account kind: {self.kind}")


        class AccountRegistry:
            def __init__(self) -> None:
                self._accounts: dict[str, Account] = {}

            def register(self, account: Account) -> None:
                if account.account_id in self._accounts:
                    raise ValueError(f"duplicate account: {account.account_id}")
                self._accounts[account.account_id] = account

            def get(self, account_id: str) -> Account:
                return self._accounts[account_id]

            def all_accounts(self) -> list[Account]:
                return list(self._accounts.values())
        """
    ),
    "ledgerx/ledger.py": textwrap.dedent(
        """\
        from ledgerx.accounts import AccountRegistry
        from ledgerx.models import JournalEntry, UnbalancedEntryError


        class Ledger:
            def __init__(self, registry: AccountRegistry) -> None:
                self.registry = registry
                self.entries: list[JournalEntry] = []
                self._by_id: dict[str, JournalEntry] = {}

            def post_entry(self, entry: JournalEntry) -> JournalEntry:
                if entry.entry_id in self._by_id:
                    raise ValueError(f"duplicate entry: {entry.entry_id}")
                if not entry.postings:
                    raise ValueError("entry has no postings")
                for p in entry.postings:
                    self.registry.get(p.account_id)
                if sum(p.amount_cents for p in entry.postings) != 0:
                    raise UnbalancedEntryError(f"entry {entry.entry_id} does not balance")
                self.entries.append(entry)
                self._by_id[entry.entry_id] = entry
                return entry

            def balance(self, account_id: str) -> int:
                self.registry.get(account_id)
                return sum(
                    p.amount_cents
                    for e in self.entries
                    for p in e.postings
                    if p.account_id == account_id
                )

            def trial_balance(self) -> dict[str, int]:
                return {a.account_id: self.balance(a.account_id) for a in self.registry.all_accounts()}
        """
    ),
    "ledgerx/fees.py": textwrap.dedent(
        """\
        from decimal import Decimal, ROUND_HALF_UP
        from ledgerx.models import Posting


        class FeeSchedule:
            def __init__(self, flat_cents: int = 0, bps: int = 0) -> None:
                self.flat_cents = flat_cents
                self.bps = bps

            def compute_fee(self, gross_cents: int) -> int:
                bps_part = int(
                    (Decimal(gross_cents) * Decimal(self.bps) / Decimal(10000)).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                )
                return bps_part + self.flat_cents

            def make_fee_postings(
                self, gross_cents: int, payer_account_id: str, fee_revenue_account_id: str
            ) -> tuple[Posting, Posting]:
                fee = self.compute_fee(gross_cents)
                return (
                    Posting(payer_account_id, -fee),
                    Posting(fee_revenue_account_id, fee),
                )
        """
    ),
    "ledgerx/settlement.py": textwrap.dedent(
        """\
        from ledgerx.ledger import Ledger
        from ledgerx.models import JournalEntry, Posting


        class SettlementBatch:
            def net_payouts(self, ledger: Ledger, account_ids: list[str]) -> dict[str, int]:
                out = {}
                for aid in account_ids:
                    b = ledger.balance(aid)
                    if b != 0:
                        out[aid] = b
                return out

            def create_settlement_entries(
                self, ledger: Ledger, merchant_account_ids: list[str], clearing_account_id: str, batch_id: str
            ) -> list[JournalEntry]:
                created = []
                for aid in merchant_account_ids:
                    b = ledger.balance(aid)
                    if b > 0:
                        entry = JournalEntry(
                            entry_id=f"SETTLE:{batch_id}:{aid}",
                            description=f"Settlement {batch_id}",
                            postings=(Posting(aid, -b), Posting(clearing_account_id, b)),
                        )
                        ledger.post_entry(entry)
                        created.append(entry)
                return created
        """
    ),
    "ledgerx/reports.py": textwrap.dedent(
        """\
        from ledgerx.ledger import Ledger


        def income_statement(ledger: Ledger) -> dict[str, int]:
            rev = sum(ledger.balance(a.account_id) for a in ledger.registry.all_accounts() if a.kind == "revenue")
            exp = sum(ledger.balance(a.account_id) for a in ledger.registry.all_accounts() if a.kind == "expense")
            return {"revenue": rev, "expense": exp, "net_income": rev + exp}


        def balance_sheet(ledger: Ledger) -> dict[str, int]:
            res = {"asset": 0, "liability": 0, "equity": 0}
            for a in ledger.registry.all_accounts():
                if a.kind in res:
                    res[a.kind] += ledger.balance(a.account_id)
            return res


        def account_statement(ledger: Ledger, account_id: str) -> list[dict]:
            ledger.registry.get(account_id)
            running = 0
            rows = []
            for e in ledger.entries:
                delta = sum(p.amount_cents for p in e.postings if p.account_id == account_id)
                if delta != 0:
                    running += delta
                    rows.append(
                        {
                            "entry_id": e.entry_id,
                            "description": e.description,
                            "delta": delta,
                            "running_balance": running,
                        }
                    )
            return rows
        """
    ),
    "ledgerx/serialization.py": textwrap.dedent(
        """\
        import json
        from ledgerx.accounts import Account, AccountRegistry
        from ledgerx.ledger import Ledger
        from ledgerx.models import JournalEntry, Posting


        def export_json(ledger: Ledger) -> str:
            data = {
                "schema_version": 1,
                "accounts": [
                    {"account_id": a.account_id, "name": a.name, "kind": a.kind}
                    for a in ledger.registry.all_accounts()
                ],
                "entries": [
                    {
                        "entry_id": e.entry_id,
                        "description": e.description,
                        "postings": [
                            {"account_id": p.account_id, "amount_cents": p.amount_cents}
                            for p in e.postings
                        ],
                    }
                    for e in ledger.entries
                ],
            }
            return json.dumps(data, indent=2)


        def import_json(payload: str) -> Ledger:
            data = json.loads(payload)
            reg = AccountRegistry()
            for a in data["accounts"]:
                reg.register(Account(a["account_id"], a["name"], a["kind"]))
            ledger = Ledger(reg)
            for e in data["entries"]:
                postings = tuple(Posting(p["account_id"], p["amount_cents"]) for p in e["postings"])
                ledger.post_entry(JournalEntry(e["entry_id"], e["description"], postings))
            return ledger
        """
    ),
    "ledgerx/cli_helpers.py": textwrap.dedent(
        """\
        from ledgerx.ledger import Ledger


        def format_ledger_summary(ledger: Ledger) -> str:
            lines = []
            for acc in sorted(ledger.registry.all_accounts(), key=lambda a: a.account_id):
                bal = ledger.balance(acc.account_id)
                lines.append(f"{acc.account_id} ({acc.name}): {bal / 100:.2f}")
            return "\\n".join(lines)
        """
    ),
    "tests/test_v2_smoke.py": textwrap.dedent(
        """\
        import unittest
        from decimal import Decimal

        from ledgerx.accounts import Account, AccountRegistry
        from ledgerx.ledger import Ledger
        from ledgerx.models import JournalEntry, Posting
        from ledgerx.money import CurrencyMismatchError, Money


        class TestV2Smoke(unittest.TestCase):
            def test_money_basic(self):
                m = Money(1250, "USD") + Money(250, "USD")
                self.assertEqual(m, Money(1500, "USD"))
                self.assertEqual(m.format(), "USD 15.00")
                with self.assertRaises(CurrencyMismatchError):
                    _ = Money(100, "USD") + Money(100, "EUR")

            def test_ledger_post_and_balance(self):
                reg = AccountRegistry()
                reg.register(Account("cash_usd", "Cash", "asset", "USD"))
                reg.register(Account("rev_usd", "Revenue", "revenue", "USD"))
                led = Ledger(reg)
                led.post_entry(
                    JournalEntry(
                        "E1",
                        "Sale",
                        (Posting("cash_usd", Money(-5000, "USD")), Posting("rev_usd", Money(5000, "USD"))),
                    )
                )
                self.assertEqual(led.balance("rev_usd"), Money(5000, "USD"))


        if __name__ == "__main__":
            unittest.main()
        """
    ),
}

LEDGERX_HIDDEN_TESTS = textwrap.dedent(
    """\
    import json
    import unittest
    from decimal import Decimal

    from ledgerx.accounts import Account, AccountRegistry
    from ledgerx.cli_helpers import format_ledger_summary
    from ledgerx.fees import FeeSchedule
    from ledgerx.ledger import Ledger
    from ledgerx.models import AlreadyVoidedError, JournalEntry, Posting, UnbalancedEntryError
    from ledgerx.money import CurrencyMismatchError, Money
    from ledgerx.reports import account_statement, balance_sheet, income_statement
    from ledgerx.serialization import export_json, import_json
    from ledgerx.settlement import SettlementBatch


    def _make_ledger():
        reg = AccountRegistry()
        reg.register(Account("cash_usd", "Cash USD", "asset", "USD"))
        reg.register(Account("cash_eur", "Cash EUR", "asset", "EUR"))
        reg.register(Account("merch_usd", "Merchant USD", "liability", "USD"))
        reg.register(Account("merch_eur", "Merchant EUR", "liability", "EUR"))
        reg.register(Account("clear_usd", "Clearing USD", "asset", "USD"))
        reg.register(Account("clear_eur", "Clearing EUR", "asset", "EUR"))
        reg.register(Account("rev_usd", "Fee Rev USD", "revenue", "USD"))
        reg.register(Account("exp_usd", "Ops Exp USD", "expense", "USD"))
        reg.register(Account("eq_usd", "Equity USD", "equity", "USD"))
        return Ledger(reg)


    class TestLedgerXV2Hidden(unittest.TestCase):
        def test_01_money_validation_and_format(self):
            with self.assertRaises(TypeError):
                Money(True, "USD")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                Money(1.5, "USD")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                Money(100, "usd")
            with self.assertRaises(ValueError):
                Money(100, "US")
            self.assertEqual((-Money(505, "EUR")).format(), "EUR -5.05")
            self.assertEqual(Money.zero("GBP"), Money(0, "GBP"))
            self.assertNotEqual(Money(0, "USD"), Money(0, "EUR"))

        def test_02_money_comparisons(self):
            self.assertTrue(Money(100, "USD") < Money(200, "USD"))
            self.assertTrue(Money(200, "USD") >= Money(200, "USD"))
            with self.assertRaises(CurrencyMismatchError):
                _ = Money(100, "USD") < Money(200, "EUR")

        def test_03_posting_convenience_properties(self):
            p = Posting("cash_usd", Money(420, "USD"))
            self.assertEqual(p.amount_cents, 420)
            self.assertEqual(p.currency, "USD")

        def test_04_account_currency_filter(self):
            led = _make_ledger()
            eur_ids = {a.account_id for a in led.registry.all_accounts(currency="EUR")}
            self.assertEqual(eur_ids, {"cash_eur", "merch_eur", "clear_eur"})

        def test_05_posting_account_currency_mismatch(self):
            led = _make_ledger()
            with self.assertRaises(CurrencyMismatchError):
                led.post_entry(
                    JournalEntry(
                        "E_BAD",
                        "wrong currency",
                        (Posting("cash_usd", Money(-100, "EUR")), Posting("merch_eur", Money(100, "EUR"))),
                    )
                )

        def test_06_multi_currency_same_entry_balanced(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry(
                    "E_MULTI",
                    "Both USD and EUR balanced independently",
                    (
                        Posting("cash_usd", Money(-1000, "USD")),
                        Posting("merch_usd", Money(1000, "USD")),
                        Posting("cash_eur", Money(-500, "EUR")),
                        Posting("merch_eur", Money(500, "EUR")),
                    ),
                )
            )
            self.assertEqual(led.balance("merch_usd"), Money(1000, "USD"))
            self.assertEqual(led.balance("merch_eur"), Money(500, "EUR"))

        def test_07_multi_currency_unbalanced_rejected(self):
            led = _make_ledger()
            with self.assertRaises(UnbalancedEntryError):
                led.post_entry(
                    JournalEntry(
                        "E_UNBAL",
                        "USD +100, EUR -100 without FX",
                        (Posting("cash_usd", Money(100, "USD")), Posting("cash_eur", Money(-100, "EUR"))),
                    )
                )

        def test_08_fx_entry_conversion_and_inverse(self):
            led = _make_ledger()
            # 1 EUR = 1.10 USD => 1000 EUR cents = 1100 USD cents
            led.post_entry(
                JournalEntry(
                    "E_FX1",
                    "FX conversion",
                    (Posting("cash_eur", Money(-1000, "EUR")), Posting("cash_usd", Money(1100, "USD"))),
                ),
                fx_rates={("EUR", "USD"): Decimal("1.10")},
                base_currency="USD",
            )
            # Test inverse lookup: provide ("USD", "EUR") = 0.5 => 1 EUR = 2.0 USD
            led.post_entry(
                JournalEntry(
                    "E_FX2",
                    "FX inverse",
                    (Posting("cash_eur", Money(-500, "EUR")), Posting("cash_usd", Money(1000, "USD"))),
                ),
                fx_rates={("USD", "EUR"): Decimal("0.5")},
                base_currency="USD",
            )
            self.assertEqual(led.balance("cash_eur"), Money(-1500, "EUR"))
            self.assertEqual(led.balance("cash_usd"), Money(2100, "USD"))

        def test_09_fx_entry_missing_base_or_unbalanced(self):
            led = _make_ledger()
            with self.assertRaises(ValueError):
                led.post_entry(
                    JournalEntry("E_NOBASE", "x", (Posting("cash_eur", Money(-100, "EUR")), Posting("cash_usd", Money(110, "USD")))),
                    fx_rates={("EUR", "USD"): Decimal("1.10")},
                )
            with self.assertRaises(UnbalancedEntryError):
                led.post_entry(
                    JournalEntry("E_FXBAD", "x", (Posting("cash_eur", Money(-100, "EUR")), Posting("cash_usd", Money(105, "USD")))),
                    fx_rates={("EUR", "USD"): Decimal("1.10")},
                    base_currency="USD",
                )

        def test_10_void_entry_and_double_void(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry(
                    "E1",
                    "Sale",
                    (Posting("cash_usd", Money(-2500, "USD")), Posting("merch_usd", Money(2500, "USD"))),
                )
            )
            rev = led.void_entry("E1", "customer cancelled")
            self.assertEqual(rev.entry_id, "E1:VOID")
            self.assertEqual(rev.void_of, "E1")
            self.assertEqual(rev.void_reason, "customer cancelled")
            self.assertEqual(led.balance("merch_usd"), Money(0, "USD"))
            with self.assertRaises(AlreadyVoidedError):
                led.void_entry("E1", "again")
            with self.assertRaises(AlreadyVoidedError):
                led.void_entry("E1:VOID", "cannot void a void")

        def test_11_trial_balance_filtered(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry("E1", "x", (Posting("cash_eur", Money(-300, "EUR")), Posting("merch_eur", Money(300, "EUR"))))
            )
            tb_eur = led.trial_balance(currency="EUR")
            self.assertEqual(set(tb_eur.keys()), {"cash_eur", "merch_eur", "clear_eur"})
            self.assertEqual(tb_eur["merch_eur"], Money(300, "EUR"))

        def test_12_fee_schedule_multicurrency(self):
            fs = FeeSchedule(flat_fees={"USD": Money(30, "USD"), "EUR": Money(25, "EUR")}, bps=250)
            self.assertEqual(fs.compute_fee(Money(1000, "USD")), Money(55, "USD"))
            self.assertEqual(fs.compute_fee(Money(1000, "EUR")), Money(50, "EUR"))
            self.assertEqual(fs.compute_fee(Money(1000, "GBP")), Money(25, "GBP"))
            p1, p2 = fs.make_fee_postings(Money(1000, "USD"), "merch_usd", "rev_usd")
            self.assertEqual(p1, Posting("merch_usd", Money(-55, "USD")))
            self.assertEqual(p2, Posting("rev_usd", Money(55, "USD")))

        def test_13_settlement_batch_multicurrency(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry(
                    "E1",
                    "sales",
                    (
                        Posting("cash_usd", Money(-4000, "USD")),
                        Posting("merch_usd", Money(4000, "USD")),
                        Posting("cash_eur", Money(-2000, "EUR")),
                        Posting("merch_eur", Money(2000, "EUR")),
                    ),
                )
            )
            sb = SettlementBatch()
            payouts = sb.net_payouts(led, ["merch_usd", "merch_eur", "rev_usd"])
            self.assertEqual(payouts, {"USD": {"merch_usd": Money(4000, "USD")}, "EUR": {"merch_eur": Money(2000, "EUR")}})
            created = sb.create_settlement_entries(
                led, ["merch_usd", "merch_eur"], {"USD": "clear_usd", "EUR": "clear_eur"}, "B01"
            )
            self.assertEqual(len(created), 2)
            self.assertEqual(led.balance("merch_usd"), Money(0, "USD"))
            self.assertEqual(led.balance("clear_eur"), Money(2000, "EUR"))

        def test_14_settlement_missing_clearing_currency(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry("E1", "eur", (Posting("cash_eur", Money(-500, "EUR")), Posting("merch_eur", Money(500, "EUR"))))
            )
            sb = SettlementBatch()
            with self.assertRaises(CurrencyMismatchError):
                sb.create_settlement_entries(led, ["merch_eur"], {"USD": "clear_usd"}, "B02")

        def test_15_reports_income_and_balance_sheet(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry(
                    "E1",
                    "ops",
                    (
                        Posting("cash_usd", Money(-800, "USD")),
                        Posting("rev_usd", Money(1000, "USD")),
                        Posting("exp_usd", Money(-200, "USD")),
                    ),
                )
            )
            inc = income_statement(led, "USD")
            self.assertEqual(inc, {"revenue": Money(1000, "USD"), "expense": Money(-200, "USD"), "net_income": Money(800, "USD")})
            bs = balance_sheet(led, "USD")
            self.assertEqual(bs["asset"], Money(-800, "USD"))

        def test_16_account_statement_with_void(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry("E1", "sale", (Posting("cash_usd", Money(-500, "USD")), Posting("merch_usd", Money(500, "USD"))))
            )
            led.void_entry("E1", "err")
            stmt = account_statement(led, "merch_usd")
            self.assertEqual(len(stmt), 2)
            self.assertTrue(stmt[0]["voided"])
            self.assertEqual(stmt[0]["running_balance"], Money(500, "USD"))
            self.assertFalse(stmt[1]["voided"])
            self.assertEqual(stmt[1]["running_balance"], Money(0, "USD"))

        def test_17_serialization_v2_roundtrip_and_v1_import(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry("E_FX", "fx", (Posting("cash_eur", Money(-1000, "EUR")), Posting("cash_usd", Money(1100, "USD")))),
                fx_rates={("EUR", "USD"): Decimal("1.10")},
                base_currency="USD",
            )
            led.void_entry("E_FX", "test")
            raw = export_json(led)
            parsed = json.loads(raw)
            self.assertEqual(parsed["schema_version"], 2)
            led2 = import_json(raw)
            self.assertEqual(led2.balance("cash_eur"), Money(0, "EUR"))
            self.assertTrue(led2.entries[0].voided)
            self.assertEqual(led2.entries[1].void_of, "E_FX")

            v1_payload = json.dumps(
                {
                    "schema_version": 1,
                    "accounts": [
                        {"account_id": "a1", "name": "Cash", "kind": "asset"},
                        {"account_id": "r1", "name": "Rev", "kind": "revenue"},
                    ],
                    "entries": [
                        {
                            "entry_id": "OLD1",
                            "description": "Legacy",
                            "postings": [{"account_id": "a1", "amount_cents": -750}, {"account_id": "r1", "amount_cents": 750}],
                        }
                    ],
                }
            )
            led_v1 = import_json(v1_payload)
            self.assertEqual(led_v1.balance("r1"), Money(750, "USD"))

        def test_18_cli_summary_format(self):
            led = _make_ledger()
            led.post_entry(
                JournalEntry("E1", "x", (Posting("cash_eur", Money(-105, "EUR")), Posting("merch_eur", Money(105, "EUR"))))
            )
            out = format_ledger_summary(led)
            lines = out.splitlines()
            self.assertIn("cash_eur (Cash EUR): EUR -1.05", lines)
            self.assertIn("merch_eur (Merchant EUR): EUR 1.05", lines)
            # EUR accounts must appear before USD accounts
            eur_idx = [i for i, l in enumerate(lines) if "EUR " in l]
            usd_idx = [i for i, l in enumerate(lines) if "USD " in l]
            self.assertTrue(max(eur_idx) < min(usd_idx))


    if __name__ == "__main__":
        unittest.main()
    """
)


class RefactorMultifile(ExtendedFileTools):
    name = "refactor_multifile"
    category = "SWE: 30+ turn multi-file refactor"
    max_turns = 50

    system = textwrap.dedent(
        """\
        You are a senior software engineering agent refactoring a multi-file Python repository (`ledgerx`).
        Read MIGRATION_SPEC.md carefully and inspect every module in `ledgerx/`.
        Migrate ALL modules to the v2 specification cleanly and consistently (standard library only).
        A minimal smoke test exists in `tests/test_v2_smoke.py`, and a comprehensive 18-test hidden suite
        will validate every single requirement in MIGRATION_SPEC.md across all 9 modules.
        Use `run_tests` and `run_python` to thoroughly test your own code before calling `finish(summary)`.
        """
    )
    user = "Migrate the entire `ledgerx` package from v1 to v2 according to MIGRATION_SPEC.md."

    def tools(self) -> list[dict]:
        return self.file_tools() + [
            _tool("run_tests", "Run unittest discovery on tests/.", {}, []),
            _tool("run_python", "Run a Python snippet in the workspace and return stdout/stderr.", {"code": S}),
            _tool("finish", "Declare the refactoring complete.", {"summary": S}),
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        (self.sandbox / "MIGRATION_SPEC.md").write_text(LEDGERX_MIGRATION_SPEC)
        for rel, txt in LEDGERX_V1_FILES.items():
            p = self.sandbox / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)

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
        (hidden / "test_v2_hidden.py").write_text(
            "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            + LEDGERX_HIDDEN_TESTS
        )
        out = self._run([sys.executable, ".hidden/test_v2_hidden.py", "-v"])
        ok = len(re.findall(r"\.\.\. ok", out))
        total = 18
        fails = [l for l in out.splitlines() if "FAIL:" in l or "ERROR:" in l]
        return {
            "success": ok == total,
            "score": ok / total,
            "detail": f"hidden {ok}/{total}" + (f" | fails={fails[:5]}" if fails else ""),
        }


# ===========================================================================
# TIER-2 TASK 2: Long-Context Needle-in-a-Haystack Debugging (`pipeline`)
# ===========================================================================

PIPELINE_README = textwrap.dedent(
    """\
    # StreamPulse Event Pipeline Incident (`INC-4092`)

    In production, StreamPulse (`pipeline/`) processed a 2-hour traffic window and silently lost
    customer billing events. Customers reported missing events, but NO unhandled exceptions were raised.

    ## Workspace contents
    - `pipeline/`: 6-module event ingestion pipeline (`config.py`, `dedup.py`, `window.py`, `sink.py`, `runner.py`, `__init__.py`).
    - `configs/prod_snapshot.json`: Exact production config active during the incident.
    - `logs/node_a.log`, `logs/node_b.log`, `logs/gateway.log`: Full production logs (~1,800 lines total,
      mostly healthy traffic and harmless noise). Somewhere in these logs are the exact forensic clues
      showing WHY events were dropped.
    - `tests/test_smoke.py`: Basic happy-path smoke test (currently passes!).

    ## Your job
    1. Investigate the production logs and config to find ALL root causes of silent event loss.
       (Note: there are multiple independent root causes across different modules!)
    2. Fix the bugs in `pipeline/` (do NOT weaken tests or alter `configs/prod_snapshot.json`).
    3. Verify with `run_tests` and `run_python`, then call `finish(summary)` listing the root causes you found.
    """
)

PIPELINE_FILES_BUGGY = {
    "pipeline/__init__.py": "",
    "pipeline/config.py": textwrap.dedent(
        """\
        import json
        from dataclasses import dataclass
        from pathlib import Path


        @dataclass(frozen=True)
        class PipelineConfig:
            late_arrival_grace_ms: int = 5000
            dedup_scope: str = "per_tenant"
            max_batch_bytes: int = 1024
            flush_on_close: bool = True

            @classmethod
            def from_file(cls, path: str | Path) -> "PipelineConfig":
                data = json.loads(Path(path).read_text())
                return cls(
                    late_arrival_grace_ms=int(data.get("late_arrival_grace_ms", 5000)),
                    dedup_scope=str(data.get("dedup_scope", "per_tenant")),
                    max_batch_bytes=int(data.get("max_batch_bytes", 1024)),
                    flush_on_close=bool(data.get("flush_on_close", True)),
                )
        """
    ),
    "pipeline/window.py": textwrap.dedent(
        """\
        class WatermarkFilter:
            \"\"\"Accepts events whose event_ts is within late_arrival_grace_ms of the high-water mark.\"\"\"

            def __init__(self, grace_ms: int) -> None:
                self.grace_ms = grace_ms
                self.watermark_ms: int = 0
                self.dropped_late: int = 0

            def accept(self, event_ts_ms: int) -> bool:
                if event_ts_ms > self.watermark_ms:
                    self.watermark_ms = event_ts_ms
                    return True
                # BUG 1: strict `<` drops events right on the grace boundary (watermark - event_ts == grace_ms)
                if self.watermark_ms - event_ts_ms < self.grace_ms:
                    return True
                self.dropped_late += 1
                return False
        """
    ),
    "pipeline/dedup.py": textwrap.dedent(
        """\
        class Deduplicator:
            \"\"\"Drops duplicate events according to dedup_scope ('global' or 'per_tenant').\"\"\"

            def __init__(self, scope: str = "per_tenant") -> None:
                self.scope = scope
                self._seen: set = set()
                self.dropped_dup: int = 0

            def is_duplicate(self, tenant_id: str, event_id: str) -> bool:
                # BUG 2: ignores self.scope and keys only by event_id globally!
                key = event_id
                if key in self._seen:
                    self.dropped_dup += 1
                    return True
                self._seen.add(key)
                return False
        """
    ),
    "pipeline/sink.py": textwrap.dedent(
        """\
        import json


        class BatchSink:
            \"\"\"Buffers serialized events into byte-bounded batches before writing.\"\"\"

            def __init__(self, max_batch_bytes: int, flush_on_close: bool = True) -> None:
                self.max_batch_bytes = max_batch_bytes
                self.flush_on_close = flush_on_close
                self._buf: list[dict] = []
                self._buf_bytes: int = 0
                self.flushed_batches: list[list[dict]] = []

            def write(self, event: dict) -> None:
                size = len(json.dumps(event, separators=(",", ":")).encode("utf-8"))
                if self._buf and self._buf_bytes + size > self.max_batch_bytes:
                    self.flush()
                self._buf.append(event)
                self._buf_bytes += size
                if self._buf_bytes >= self.max_batch_bytes:
                    self.flush()

            def flush(self) -> None:
                if self._buf:
                    self.flushed_batches.append(list(self._buf))
                    self._buf.clear()
                    self._buf_bytes = 0

            def close(self) -> None:
                # BUG 3: forgets to flush remaining buffered events when close() is called!
                self._buf.clear()
                self._buf_bytes = 0
        """
    ),
    "pipeline/runner.py": textwrap.dedent(
        """\
        from pipeline.config import PipelineConfig
        from pipeline.dedup import Deduplicator
        from pipeline.sink import BatchSink
        from pipeline.window import WatermarkFilter


        def run_pipeline(events: list[dict], config: PipelineConfig) -> dict:
            wf = WatermarkFilter(config.late_arrival_grace_ms)
            dedup = Deduplicator(config.dedup_scope)
            sink = BatchSink(config.max_batch_bytes, flush_on_close=config.flush_on_close)

            for ev in events:
                if not wf.accept(int(ev["ts_ms"])):
                    continue
                if dedup.is_duplicate(str(ev["tenant_id"]), str(ev["event_id"])):
                    continue
                sink.write(ev)

            sink.close()
            emitted = [ev for batch in sink.flushed_batches for ev in batch]
            return {
                "emitted": emitted,
                "batches": sink.flushed_batches,
                "dropped_late": wf.dropped_late,
                "dropped_dup": dedup.dropped_dup,
            }
        """
    ),
    "configs/prod_snapshot.json": json.dumps(
        {
            "cluster": "streampulse-us-west-2",
            "late_arrival_grace_ms": 5000,
            "dedup_scope": "per_tenant",
            "max_batch_bytes": 512,
            "flush_on_close": True,
            "metrics_interval_s": 30,
            "retry_backoff_ms": 250,
        },
        indent=2,
    ),
    "tests/test_smoke.py": textwrap.dedent(
        """\
        import unittest
        from pipeline.config import PipelineConfig
        from pipeline.runner import run_pipeline


        class TestSmoke(unittest.TestCase):
            def test_basic_flush(self):
                cfg = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="per_tenant", max_batch_bytes=100)
                events = [
                    {"tenant_id": "t1", "event_id": "e1", "ts_ms": 10000, "payload": "x" * 60},
                    {"tenant_id": "t1", "event_id": "e2", "ts_ms": 11000, "payload": "y" * 60},
                ]
                res = run_pipeline(events, cfg)
                self.assertEqual(len(res["emitted"]), 2)


        if __name__ == "__main__":
            unittest.main()
        """
    ),
}


def _strip_bug_comments(src: str) -> str:
    """Remove `# BUG ...` lines so the agent must discover bugs from logs, not comments."""
    return "\n".join(l for l in src.splitlines() if "# BUG" not in l) + "\n"


def _generate_haystack_logs() -> dict[str, str]:
    """Generate ~1,800 lines (~28k tokens) of realistic noisy logs with 3 buried needles."""
    rng = random.Random(4092)
    tenants = ["orion", "helios", "vega", "cygnus", "lyra", "draco", "altair"]
    regions = ["us-w2", "us-e1", "eu-c1"]

    def make_noise(node: str, count: int, base_ts: int) -> list[str]:
        lines = []
        ts = base_ts
        for i in range(count):
            ts += rng.randint(15, 140)
            t = rng.choice(tenants)
            r = rng.choice(regions)
            eid = f"evt-{rng.randint(100000, 999999)}"
            kind = rng.random()
            if kind < 0.72:
                lines.append(
                    f"2026-03-18T14:{(i//60)%60:02d}:{i%60:02d}.{ts%1000:03d}Z INFO  [{node}] "
                    f"ingest tenant={t} region={r} event_id={eid} ts_ms={ts} bytes={rng.randint(64, 180)} status=accepted"
                )
            elif kind < 0.86:
                lines.append(
                    f"2026-03-18T14:{(i//60)%60:02d}:{i%60:02d}.{ts%1000:03d}Z INFO  [{node}] "
                    f"metrics active_conns={rng.randint(12, 48)} queue_depth={rng.randint(0, 9)} cpu_pct={rng.uniform(18, 42):.1f}"
                )
            elif kind < 0.94:
                # Harmless red-herring warning
                lines.append(
                    f"2026-03-18T14:{(i//60)%60:02d}:{i%60:02d}.{ts%1000:03d}Z WARN  [{node}] "
                    f"tls_session_cache soft_miss tenant={t} region={r} fallback=full_handshake latency_ms={rng.randint(4, 19)} (benign)"
                )
            else:
                lines.append(
                    f"2026-03-18T14:{(i//60)%60:02d}:{i%60:02d}.{ts%1000:03d}Z WARN  [{node}] "
                    f"deprecated_metric name=sp_legacy_qps note='scheduled for removal in v4; no traffic impact'"
                )
        return lines

    # Log 1: node_a.log (~620 lines) with Needle 1 at line ~412 (no giveaway tag)
    a_lines = make_noise("node-a", 620, 1773842400000)
    a_lines[410:414] = [
        "2026-03-18T14:38:11.000Z INFO  [node-a] watermark_advance prev_wm=1773843000000 new_wm=1773843005000 grace_ms=5000",
        "2026-03-18T14:38:11.012Z INFO  [node-a] filter_decision module=window tenant=cygnus event_id=evt-771004 "
        "ts_ms=1773843000000 watermark_ms=1773843005000 lag_ms=5000 grace_ms=5000 action=drop",
        "2026-03-18T14:38:11.019Z INFO  [node-a] filter_decision module=window tenant=cygnus event_id=evt-771005 "
        "ts_ms=1773843000000 watermark_ms=1773843005000 lag_ms=5000 grace_ms=5000 action=drop",
        "2026-03-18T14:38:11.040Z INFO  [node-a] ticket_note=INC-4092-A customer=cygnus sla='events with lag_ms <= grace_ms must be accepted' missing=evt-771004,evt-771005",
    ]

    # Log 2: node_b.log (~620 lines) with Needle 2 at line ~495 (no giveaway tag)
    b_lines = make_noise("node-b", 620, 1773842500000)
    b_lines[492:497] = [
        "2026-03-18T14:44:02.110Z INFO  [node-b] ingest tenant=orion region=us-w2 event_id=evt-90412 ts_ms=1773843242110 bytes=118 status=accepted",
        "2026-03-18T14:44:05.304Z INFO  [node-b] dedup_decision module=dedup tenant=helios region=eu-c1 event_id=evt-90412 "
        "action=drop prior_tenant=orion configured_scope=per_tenant",
        "2026-03-18T14:44:05.311Z INFO  [node-b] ticket_note=INC-4092-B customer=helios missing=evt-90412",
        "2026-03-18T14:44:05.400Z INFO  [node-b] dedup_decision module=dedup tenant=orion region=us-w2 event_id=evt-90412 action=drop prior_tenant=orion",
        "2026-03-18T14:44:05.450Z INFO  [node-b] metrics dedup_table_size=14208",
    ]

    # Log 3: gateway.log (~580 lines) with Needle 3 at line ~518 (no giveaway tag)
    g_lines = make_noise("gateway", 580, 1773842600000)
    g_lines[515:520] = [
        "2026-03-18T14:59:58.900Z INFO  [gateway] sink_write event_id=evt-998801 bytes=142 buf_bytes=142 max_batch_bytes=512",
        "2026-03-18T14:59:59.105Z INFO  [gateway] sink_write event_id=evt-998802 bytes=138 buf_bytes=280 max_batch_bytes=512",
        "2026-03-18T14:59:59.990Z INFO  [gateway] shutdown_begin flush_on_close=true pending_events=2 pending_bytes=280",
        "2026-03-18T15:00:00.002Z INFO  [gateway] shutdown_complete flushed_batches_during_close=0 discarded_events=2 lost_ids=evt-998801,evt-998802",
        "2026-03-18T15:00:00.010Z INFO  [gateway] ticket_note=INC-4092-C missing=evt-998801,evt-998802",
    ]

    return {
        "logs/node_a.log": "\n".join(a_lines) + "\n",
        "logs/node_b.log": "\n".join(b_lines) + "\n",
        "logs/gateway.log": "\n".join(g_lines) + "\n",
    }


PIPELINE_HIDDEN_TESTS = textwrap.dedent(
    """\
    import unittest
    from pipeline.config import PipelineConfig
    from pipeline.runner import run_pipeline


    class TestIncidentReplay(unittest.TestCase):
        def test_01_grace_boundary_inclusive(self):
            cfg = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="per_tenant", max_batch_bytes=512, flush_on_close=True)
            events = [
                {"tenant_id": "cygnus", "event_id": "e1", "ts_ms": 10005000, "payload": "a"},
                # Exactly on grace boundary (lag == 5000): MUST be accepted
                {"tenant_id": "cygnus", "event_id": "e2", "ts_ms": 10000000, "payload": "b"},
                # Strictly past grace boundary (lag == 5001): MUST be dropped
                {"tenant_id": "cygnus", "event_id": "e3", "ts_ms": 9999999, "payload": "c"},
            ]
            res = run_pipeline(events, cfg)
            ids = [e["event_id"] for e in res["emitted"]]
            self.assertEqual(ids, ["e1", "e2"])
            self.assertEqual(res["dropped_late"], 1)

        def test_02_per_tenant_dedup_vs_global(self):
            cfg_pt = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="per_tenant", max_batch_bytes=512, flush_on_close=True)
            events = [
                {"tenant_id": "orion", "event_id": "evt-90412", "ts_ms": 10000, "payload": "1"},
                {"tenant_id": "helios", "event_id": "evt-90412", "ts_ms": 10001, "payload": "2"},
                {"tenant_id": "orion", "event_id": "evt-90412", "ts_ms": 10002, "payload": "dup"},
            ]
            res = run_pipeline(events, cfg_pt)
            pairs = [(e["tenant_id"], e["event_id"]) for e in res["emitted"]]
            self.assertEqual(pairs, [("orion", "evt-90412"), ("helios", "evt-90412")])
            self.assertEqual(res["dropped_dup"], 1)

            cfg_gl = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="global", max_batch_bytes=512, flush_on_close=True)
            res_gl = run_pipeline(events, cfg_gl)
            self.assertEqual(len(res_gl["emitted"]), 1)
            self.assertEqual(res_gl["dropped_dup"], 2)

        def test_03_sink_flush_on_close(self):
            cfg = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="per_tenant", max_batch_bytes=512, flush_on_close=True)
            events = [
                {"tenant_id": "t1", "event_id": "tail-1", "ts_ms": 20000, "payload": "short1"},
                {"tenant_id": "t1", "event_id": "tail-2", "ts_ms": 20001, "payload": "short2"},
            ]
            res = run_pipeline(events, cfg)
            self.assertEqual([e["event_id"] for e in res["emitted"]], ["tail-1", "tail-2"])

            # Respect flush_on_close=False as well
            cfg_no = PipelineConfig(late_arrival_grace_ms=5000, dedup_scope="per_tenant", max_batch_bytes=512, flush_on_close=False)
            res_no = run_pipeline(events, cfg_no)
            self.assertEqual(res_no["emitted"], [])

        def test_04_combined_full_incident_replay(self):
            cfg = PipelineConfig.from_file("configs/prod_snapshot.json")
            events = [
                {"tenant_id": "cygnus", "event_id": "evt-771000", "ts_ms": 1773843005000, "payload": "ok"},
                {"tenant_id": "cygnus", "event_id": "evt-771004", "ts_ms": 1773843000000, "payload": "boundary_ok"},
                {"tenant_id": "cygnus", "event_id": "evt-771009", "ts_ms": 1773842999999, "payload": "too_late"},
                {"tenant_id": "orion", "event_id": "evt-90412", "ts_ms": 1773843242110, "payload": "orion_ev"},
                {"tenant_id": "helios", "event_id": "evt-90412", "ts_ms": 1773843245304, "payload": "helios_ev"},
                {"tenant_id": "orion", "event_id": "evt-90412", "ts_ms": 1773843245400, "payload": "orion_dup"},
                {"tenant_id": "vega", "event_id": "evt-998801", "ts_ms": 1773843598900, "payload": "tail1"},
                {"tenant_id": "vega", "event_id": "evt-998802", "ts_ms": 1773843599105, "payload": "tail2"},
            ]
            res = run_pipeline(events, cfg)
            got = [(e["tenant_id"], e["event_id"]) for e in res["emitted"]]
            self.assertEqual(
                got,
                [
                    ("cygnus", "evt-771000"),
                    ("cygnus", "evt-771004"),
                    ("orion", "evt-90412"),
                    ("helios", "evt-90412"),
                    ("vega", "evt-998801"),
                    ("vega", "evt-998802"),
                ],
            )
            self.assertEqual(res["dropped_late"], 1)
            self.assertEqual(res["dropped_dup"], 1)


    if __name__ == "__main__":
        unittest.main()
    """
)


class NeedleDebugLongCtx(ExtendedFileTools):
    name = "needle_debug_longctx"
    category = "Long-context needle debugging"
    max_turns = 35

    system = textwrap.dedent(
        """\
        You are an incident-response engineering agent debugging a production event-loss incident (`INC-4092`).
        Read README.md, inspect `configs/prod_snapshot.json`, sift through the large production logs in `logs/`
        (over 1,800 lines of mostly benign traffic and red-herring warnings), identify EVERY root cause of
        event loss, and patch the affected modules in `pipeline/`.
        Verify your fixes with `run_tests` and `run_python`, then call `finish(summary)`.
        """
    )
    user = (
        "Investigate INC-4092 in this workspace. Find all root causes of silent event loss in the production "
        "logs, fix the bugs in `pipeline/`, and verify your fixes."
    )

    def tools(self) -> list[dict]:
        return self.file_tools() + [
            _tool("run_tests", "Run unittest discovery on tests/.", {}, []),
            _tool("run_python", "Run a Python snippet in the workspace and return stdout/stderr.", {"code": S}),
            _tool("finish", "Declare the incident investigation and fixes complete.", {"summary": S}),
        ]

    def setup(self, sandbox: Path) -> None:
        super().setup(sandbox)
        (self.sandbox / "README.md").write_text(PIPELINE_README)
        for rel, txt in PIPELINE_FILES_BUGGY.items():
            p = self.sandbox / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_strip_bug_comments(txt) if rel.startswith("pipeline/") else txt)
        for rel, txt in _generate_haystack_logs().items():
            p = self.sandbox / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)

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
        (hidden / "test_incident.py").write_text(
            "import os, sys\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
            + PIPELINE_HIDDEN_TESTS
        )
        out = self._run([sys.executable, ".hidden/test_incident.py", "-v"])
        ok = len(re.findall(r"\.\.\. ok", out))
        total = 4
        fails = [l for l in out.splitlines() if "FAIL:" in l or "ERROR:" in l]
        return {
            "success": ok == total,
            "score": ok / total,
            "detail": f"hidden {ok}/{total}" + (f" | fails={fails[:4]}" if fails else ""),
        }


class NeedlePureLongCtx(NeedleDebugLongCtx):
    """Same 3-bug incident, but the full ~20k-token log haystack is injected directly into the
    initial user context (no `search` tool) to stress true long-context KV-cache attention recall."""

    name = "needle_pure_longctx"
    category = "Pure 20k+ long-context needle debugging"
    max_turns = 24

    def __init__(self) -> None:
        super().__init__()
        logs = _generate_haystack_logs()
        # ~150 lines per file (~450 lines total => ~24k-26k tokens) with needles at ~25%, ~55%, ~85%
        parts = []
        for fname, content in logs.items():
            lines = content.splitlines()
            # Needles are at indices 410..413 (node_a), 492..496 (node_b), 515..519 (gateway)
            if "node_a" in fname:
                kept = lines[310:460]  # needle at ~100/150
            elif "node_b" in fname:
                kept = lines[410:560]  # needle at ~82/150
            else:
                kept = lines[410:560]  # needle at ~105/150
            parts.append(f"=== {fname} ===\n" + "\n".join(kept))
        bundle = "\n\n".join(parts)
        self.user = (
            "Below is the full production log dump for incident INC-4092 (there is no `search` tool; "
            "all forensic clues are in these logs). Identify ALL 3 root causes of silent event loss, "
            "patch `pipeline/`, verify with `run_tests` / `run_python`, and call `finish(summary)`.\n\n"
            + bundle
        )

    def tools(self) -> list[dict]:
        return [
            t for t in self.file_tools() if t["function"]["name"] != "search"
        ] + [
            _tool("run_tests", "Run unittest discovery on tests/.", {}, []),
            _tool("run_python", "Run a Python snippet in the workspace and return stdout/stderr.", {"code": S}),
            _tool("finish", "Declare the incident investigation and fixes complete.", {"summary": S}),
        ]


def tier2_tasks() -> list[Any]:
    return [
        RefactorMultifile(),
        NeedleDebugLongCtx(),
        NeedlePureLongCtx(),
    ]


def selftest() -> None:
    import tempfile

    # 1. Verify needle_debug_longctx fails when unpatched and passes when patched
    with tempfile.TemporaryDirectory() as d:
        t = NeedleDebugLongCtx()
        t.setup(Path(d))
        r0 = t.check("")
        assert not r0["success"], f"Expected buggy pipeline to fail, got {r0}"
        t.t_replace_in_file("pipeline/window.py", "< self.grace_ms:", "<= self.grace_ms:")
        t.t_replace_in_file(
            "pipeline/dedup.py",
            "key = event_id",
            'key = (tenant_id, event_id) if self.scope == "per_tenant" else event_id',
        )
        t.t_replace_in_file(
            "pipeline/sink.py",
            "self._buf.clear()\n        self._buf_bytes = 0",
            "if self.flush_on_close:\n            self.flush()\n        else:\n            self._buf.clear()\n            self._buf_bytes = 0",
        )
        r1 = t.check("")
        log_chars = sum((t.sandbox / f"logs/{n}").stat().st_size for n in ("node_a.log", "node_b.log", "gateway.log"))
        print(f"needle_debug_longctx: buggy={r0['detail']} | fixed={r1['detail']} | log_size={log_chars:,} bytes")
        assert r1["success"], r1

    # 2. Verify refactor_multifile fails on v1 codebase
    with tempfile.TemporaryDirectory() as d:
        t = RefactorMultifile()
        t.setup(Path(d))
        r0 = t.check("")
        print(f"refactor_multifile v1 baseline: {r0['detail']}")
        assert not r0["success"]

    print("tier2 selftest OK")


if __name__ == "__main__":
    selftest()
