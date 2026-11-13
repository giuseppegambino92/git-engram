"""Ground-truth developer queries over the ``shopfront`` fixture.

Relevance targets:
  ``commit:<subject prefix>``  the commit capsule whose subject starts with the prefix (grade 2 = the cause)
  ``sym:<path>::<qualname>``   an AST symbol document
  ``file:<path>``              any symbol/file document of that path

``failures`` lists Graveyard entries an ideal system must surface as "do not repeat" warnings;
``coupled`` is a file that is *only* discoverable through temporal co-change (no import edge, little lexical
overlap with the query); ``acceptable`` lists failures that are relevant enough not to count as false alarms.
Queries are paraphrased: they deliberately avoid copying commit-message wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuery:
    qid: str
    category: str
    text: str
    relevant: dict[str, int]
    failures: tuple[str, ...] = ()
    coupled: str | None = None
    acceptable: tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


C_SKEW = "commit:fix(auth): tolerate 30s clock skew"
C_IDEM = "commit:fix(payments): reuse one idempotency key"
C_PERF = "commit:perf(cache): skip the lock"
C_REVERT = 'commit:Revert "perf(cache)'
C_SEC131 = "commit:fix(security): HTML-escape customer names"
C_ACK = "commit:fix(worker): ack only after the handler"
C_BANKER = "commit:fix(money): use banker's rounding"
C_BUTTON = "commit:fix(web): disable the pay button"
C_SEC118 = "commit:fix(auth): make refresh tokens single-use"
C_JITTER = "commit:fix(worker): add full jitter"
C_NFKC = "commit:fix(search): NFKC-normalise queries"
C_STALE = "commit:fix(cart): reject checkout of outdated carts"
C_E4012 = "commit:feat(payments): friendly message for E4012"

TOKEN = "sym:backend/auth/token.py::is_token_expired"
RETRY = "sym:backend/payments/retry.py::charge_with_retry"
CACHE_GET = "sym:backend/cache/lru.py::TTLCache.get"
RENDER = "sym:backend/notifications/email.py::render_receipt"
HANDLE = "sym:worker/pkg/queue/consumer.go::Consumer.HandleOne"
TAX = "sym:backend/utils/money.py::tax_cents"
BUTTON = "sym:web/src/components/CheckoutButton.tsx::CheckoutButton"
REFRESH = "sym:backend/auth/refresh.py::refresh_session"
BACKOFF = "sym:worker/pkg/queue/backoff.go::Backoff"
NORMALIZE = "sym:backend/search/query_parser.py::normalize_query"
SKU = "sym:backend/search/query_parser.py::extract_sku"
CHECKOUT = "sym:backend/cart/service.py::CartService.checkout"
STALE_ERR = "sym:backend/cart/service.py::CartStaleError"
ORDER_SCHEMA = "sym:backend/api/schemas.py::OrderSchema"
PARSE_ORDER = "sym:mobile/src/parsers/orderParser.ts::parseOrder"
USER_MSG = "sym:backend/payments/gateway.py::user_message_for"
CHECKOUT_TS = "sym:web/src/api/cartClient.ts::checkoutCart"
ORDER_ROW = "sym:backend/db/models.py::OrderRow"

QUERIES: list[EvalQuery] = [
    # ------------------------------------------------------------------ causal "why"
    EvalQuery(
        "why-01",
        "causal",
        "Why does is_token_expired subtract a grace period before comparing with exp?",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    EvalQuery(
        "why-02",
        "causal",
        "why is the idempotency key computed once outside the loop in charge_with_retry",
        {C_IDEM: 2, RETRY: 1},
        ("F2-idempotency",),
    ),
    EvalQuery(
        "why-03",
        "causal",
        "Why does TTLCache.get take the lock even though it only reads?",
        {C_REVERT: 2, C_PERF: 1, CACHE_GET: 1},
        ("F6-cache-lock",),
    ),
    EvalQuery(
        "why-04", "causal", "why are customer names escaped when we render the receipt mail", {C_SEC131: 2, RENDER: 1}
    ),
    EvalQuery(
        "why-05",
        "causal",
        "why does the queue consumer acknowledge a message only after the handler returns",
        {C_ACK: 2, HANDLE: 1},
        ("F5-ack-order",),
    ),
    EvalQuery(
        "why-06",
        "causal",
        "reason we round sales tax half-to-even instead of half-up",
        {C_BANKER: 2, TAX: 1},
        ("F4-rounding",),
    ),
    EvalQuery("why-07", "causal", "why is the pay now button disabled after the first click", {C_BUTTON: 2, BUTTON: 1}),
    EvalQuery(
        "why-08",
        "causal",
        "why does refresh_session remember refresh tokens it has already seen",
        {C_SEC118: 2, REFRESH: 1},
    ),
    # ------------------------------------------------------------- bug / regression
    EvalQuery(
        "bug-01",
        "regression",
        "customers were billed twice after the payment provider timed out",
        {C_IDEM: 2, RETRY: 1},
        ("F2-idempotency",),
    ),
    EvalQuery(
        "bug-02",
        "regression",
        "people whose phone clock is wrong get logged out right after signing in",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    EvalQuery(
        "bug-03",
        "regression",
        "cache hands back corrupted entries and KeyError under concurrent traffic",
        {C_REVERT: 2, CACHE_GET: 1},
        ("F6-cache-lock",),
    ),
    EvalQuery(
        "bug-04",
        "regression",
        "retrying workers all hammer the broker at the same moment after an outage",
        {C_JITTER: 2, BACKOFF: 1},
    ),
    EvalQuery(
        "bug-05",
        "regression",
        "typing full-width digits in the search box does not find the product SKU",
        {C_NFKC: 2, NORMALIZE: 1, SKU: 1},
    ),
    EvalQuery(
        "bug-06",
        "regression",
        "an intercepted refresh token could be exchanged again to mint new sessions",
        {C_SEC118: 2, REFRESH: 1},
    ),
    EvalQuery(
        "bug-07",
        "regression",
        "order totals are one cent off compared with the finance ledger",
        {C_BANKER: 2, TAX: 1},
        ("F4-rounding",),
    ),
    EvalQuery(
        "bug-08",
        "regression",
        "messages disappear when the handler crashes after they were acknowledged",
        {C_ACK: 2, HANDLE: 1},
        ("F5-ack-order",),
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "cpl-01",
        "coupled",
        "add a discount_cents field to the order schema returned by the API",
        {ORDER_SCHEMA: 2, "file:mobile/src/parsers/orderParser.ts": 2, "file:web/src/api/orderTypes.ts": 1},
        coupled="mobile/src/parsers/orderParser.ts",
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "cpl-02",
        "coupled",
        "rename the currency attribute of OrderSchema",
        {ORDER_SCHEMA: 2, "file:web/src/api/orderTypes.ts": 2, "file:mobile/src/parsers/orderParser.ts": 1},
        coupled="web/src/api/orderTypes.ts",
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "cpl-03",
        "coupled",
        "change the default payment gateway timeout in the backend settings module",
        {"file:backend/config/settings.py": 2, "file:deploy/helm/values.yaml": 2},
        coupled="deploy/helm/values.yaml",
        acceptable=("F2-idempotency",),
    ),
    EvalQuery(
        "cpl-04",
        "coupled",
        "change how CartService.checkout signals a version mismatch",
        {CHECKOUT: 2, "file:web/src/api/cartClient.ts": 2},
        coupled="web/src/api/cartClient.ts",
        acceptable=("F3-asyncio",),
    ),
    EvalQuery(
        "cpl-05",
        "coupled",
        "tune the exponential Backoff delays of the queue worker",
        {BACKOFF: 2, "file:worker/config/worker.yaml": 2},
        coupled="worker/config/worker.yaml",
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "cpl-06",
        "coupled",
        "show the store address in render_receipt",
        {RENDER: 2, "file:backend/templates/receipt.html": 2},
        coupled="backend/templates/receipt.html",
    ),
    EvalQuery(
        "cpl-07",
        "coupled",
        "add a fulfilment_status column to the orders table in schema.sql",
        {"file:backend/db/schema.sql": 2, ORDER_ROW: 2, ORDER_SCHEMA: 1},
        coupled="backend/db/models.py",
    ),
    EvalQuery(
        "cpl-08",
        "coupled",
        "make parseOrder in the mobile app handle a new field",
        {PARSE_ORDER: 2, "file:backend/api/schemas.py": 2, "file:web/src/api/orderTypes.ts": 1},
        ("F7-parser-dollars",),
        coupled="backend/api/schemas.py",
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery(
        "err-01", "error-code", "ERR_CART_STALE", {C_STALE: 2, STALE_ERR: 2, CHECKOUT_TS: 1}, acceptable=("F3-asyncio",)
    ),
    EvalQuery("err-02", "error-code", "E4012", {USER_MSG: 2, C_E4012: 2}, acceptable=("F2-idempotency",)),
    EvalQuery("err-03", "error-code", "INC-2291", {C_SKEW: 2, TOKEN: 1}, acceptable=("F1-clock-skew",)),
    EvalQuery(
        "err-04", "error-code", "ERR_PAYMENT_DUPLICATE raised in production", {C_IDEM: 2, RETRY: 2}, ("F2-idempotency",)
    ),
    EvalQuery("err-05", "error-code", "SEC-131", {C_SEC131: 2, RENDER: 1}),
    EvalQuery(
        "err-06",
        "error-code",
        "RuntimeError: Event loop is closed in CartService.checkout",
        {CHECKOUT: 2},
        ("F3-asyncio",),
    ),
    EvalQuery(
        "err-07", "error-code", "WARNING: DATA RACE in Consumer.HandleOne", {HANDLE: 2, C_ACK: 1}, ("F5-ack-order",)
    ),
    EvalQuery(
        "err-08", "error-code", "what did commit {short:Revert} change", {C_REVERT: 2, CACHE_GET: 1}, ("F6-cache-lock",)
    ),
]
