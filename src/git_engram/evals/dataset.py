"""Ground-truth developer queries over the ``shopfront`` fixture.

Relevance targets:
  ``commit:<subject prefix>``  the commit capsule whose subject starts with the prefix (grade 2 = the cause)
  ``sym:<path>::<qualname>``   an AST symbol document
  ``file:<path>``              any symbol/file document of that path

``failures`` lists Graveyard entries an ideal system must surface as "do not repeat" warnings;
``coupled`` is a file that is *only* discoverable through temporal co-change (no import edge, little lexical
overlap with the query); ``acceptable`` lists failures that are relevant enough not to count as false alarms.
Queries are paraphrased: they deliberately avoid copying commit-message wording.

Two splits:

* ``DEV_QUERIES`` (32) -- the calibration split. Every weight and threshold in the engine was chosen while
  looking at these, so their scores are optimistic by construction.
* ``TEST_QUERIES`` (24) -- the held-out split, written and committed *before* the query-adaptive router existed
  and never used for any tuning decision. Many target fixture facts the dev split never asks about, and the
  coupled-file queries approach each coupling from the opposite side. Report these as the headline numbers.
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
C_CENTS = "commit:feat(api): switch order totals to integer cents"
C_TIMEOUT8 = "commit:fix(config): raise payment timeout to 8s"
C_USERDOC = "commit:docs(api): document UserSchema exposure rules"
C_SHIPTYPE = "commit:fix(web): add missing shippingCents"

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
USER_SCHEMA = "sym:backend/api/schemas.py::UserSchema"
WEB_ORDER = "sym:web/src/api/orderTypes.ts::Order"
MOBILE_ORDER = "sym:mobile/src/parsers/orderParser.ts::MobileOrder"
DECLINED = "sym:backend/payments/gateway.py::PaymentDeclined"

DEV_QUERIES: list[EvalQuery] = [
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


TEST_QUERIES: list[EvalQuery] = [
    # ------------------------------------------------------------------ causal "why"
    EvalQuery(
        "t-why-01",
        "causal",
        "what is the purpose of the CLOCK_SKEW_SECONDS constant",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    EvalQuery(
        "t-why-02",
        "causal",
        "why does checkoutCart reload the cart when the server rejects the request",
        {C_STALE: 2, CHECKOUT_TS: 1},
        acceptable=("F3-asyncio",),
    ),
    EvalQuery(
        "t-why-03",
        "causal",
        "why are order amounts stored as integer cents rather than floats",
        {C_CENTS: 2, ORDER_SCHEMA: 1},
        acceptable=("F7-parser-dollars", "F4-rounding"),
    ),
    EvalQuery(
        "t-why-04",
        "causal",
        "why was the payment gateway timeout increased to 8 seconds",
        {C_TIMEOUT8: 2, "file:backend/config/settings.py": 1},
        acceptable=("F2-idempotency",),
    ),
    EvalQuery(
        "t-why-05",
        "causal",
        "why does Backoff randomise the delay with rand.Int63n",
        {C_JITTER: 2, BACKOFF: 1},
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "t-why-06",
        "causal",
        "why must the public user schema never contain password hashes",
        {C_USERDOC: 2, USER_SCHEMA: 1},
    ),
    # ------------------------------------------------------------- bug / regression
    EvalQuery(
        "t-bug-01",
        "regression",
        "two browser tabs paid for an out-of-date list of items",
        {C_STALE: 2, CHECKOUT: 1},
        acceptable=("F3-asyncio",),
    ),
    EvalQuery(
        "t-bug-02",
        "regression",
        "a script tag in a customer display name ran inside webmail clients",
        {C_SEC131: 2, RENDER: 1},
    ),
    EvalQuery("t-bug-03", "regression", "double clicking the pay button created two orders", {C_BUTTON: 2, BUTTON: 1}),
    EvalQuery(
        "t-bug-04",
        "regression",
        "web app broke because the Order type was missing the shipping amount",
        {C_SHIPTYPE: 2, "file:web/src/api/orderTypes.ts": 1},
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "t-bug-05",
        "regression",
        "tokens issued to the mobile app looked expired the moment they were created",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    EvalQuery(
        "t-bug-06",
        "regression",
        "after removing the read lock, catalogue lookups raised KeyError under load",
        {C_REVERT: 2, CACHE_GET: 1},
        ("F6-cache-lock",),
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "t-cpl-01",
        "coupled",
        "change the CACHE_TTL_SECONDS value in the helm chart values",
        {"file:deploy/helm/values.yaml": 2, "file:backend/config/settings.py": 2},
        coupled="backend/config/settings.py",
    ),
    EvalQuery(
        "t-cpl-02",
        "coupled",
        "redesign the HTML layout of the receipt template",
        {"file:backend/templates/receipt.html": 2, RENDER: 2},
        coupled="backend/notifications/email.py",
    ),
    EvalQuery(
        "t-cpl-03",
        "coupled",
        "raise max_attempts in the worker retry configuration file",
        {"file:worker/config/worker.yaml": 2, BACKOFF: 2},
        coupled="worker/pkg/queue/backoff.go",
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "t-cpl-04",
        "coupled",
        "add a field to the OrderRow dataclass",
        {ORDER_ROW: 2, "file:backend/db/schema.sql": 2},
        coupled="backend/db/schema.sql",
    ),
    EvalQuery(
        "t-cpl-05",
        "coupled",
        "add a giftWrap flag to the web Order interface",
        {WEB_ORDER: 2, "file:backend/api/schemas.py": 2, "file:mobile/src/parsers/orderParser.ts": 1},
        coupled="backend/api/schemas.py",
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "t-cpl-06",
        "coupled",
        "change the MobileOrder type used by the mobile app",
        {MOBILE_ORDER: 2, "file:backend/api/schemas.py": 2},
        coupled="backend/api/schemas.py",
        acceptable=("F7-parser-dollars",),
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery("t-err-01", "error-code", "SEC-118", {C_SEC118: 2, REFRESH: 1}),
    EvalQuery("t-err-02", "error-code", "INC-2510", {C_REVERT: 2, CACHE_GET: 1}, acceptable=("F6-cache-lock",)),
    EvalQuery("t-err-03", "error-code", "INC-2655", {C_TIMEOUT8: 2, "file:backend/config/settings.py": 1}),
    EvalQuery(
        "t-err-04",
        "error-code",
        "KeyError: 'sku-000123' in TTLCache.get",
        {CACHE_GET: 2, C_REVERT: 1},
        ("F6-cache-lock",),
    ),
    EvalQuery(
        "t-err-05",
        "error-code",
        "PaymentDeclined E4012 expired_card",
        {USER_MSG: 2, C_E4012: 2, DECLINED: 1},
        acceptable=("F2-idempotency",),
    ),
    EvalQuery(
        "t-err-06",
        "error-code",
        "AssertionError: assert 83 == 82 in the tax rounding test",
        {TAX: 2, C_BANKER: 1},
        ("F4-rounding",),
    ),
]

QUERIES = DEV_QUERIES  # backwards-compatible alias
SPLITS = {"dev": DEV_QUERIES, "test": TEST_QUERIES}
