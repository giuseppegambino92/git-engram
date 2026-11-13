"""Ground-truth developer queries over the ``shopfront`` fixture.

Relevance targets:
  ``commit:<subject prefix>``  the commit capsule whose subject starts with the prefix (grade 2 = the cause)
  ``sym:<path>::<qualname>``   an AST symbol document
  ``file:<path>``              any symbol/file document of that path

``failures`` lists Graveyard entries an ideal system must surface as "do not repeat" warnings;
``coupled`` is a file that is *only* discoverable through temporal co-change (no import edge, little lexical
overlap with the query); ``acceptable`` lists failures that are relevant enough not to count as false alarms.
Queries are paraphrased: they deliberately avoid copying commit-message wording.

Six splits:

* ``DEV_QUERIES`` (48) -- the calibration split (8 per category, 16 natural-language bug reports). Every
  weight, threshold and design choice in the engine was made while looking at these: scores are optimistic.
* ``TEST_V1_QUERIES`` (24) -- **retired**. Committed before the query router; later used to diagnose
  commit-capsule crowding.
* ``TEST_V2_QUERIES`` (24) -- **retired**. Committed before the commit slot budget; later used to diagnose
  graveyard false alarms.
* ``TEST_V3_QUERIES`` (24) -- **retired**. Committed before symbol-level anchoring, the MMR removal and exact-route
  tuning; later used to diagnose the causal-query regression and sibling-commit false alarms. Contains
  sibling-symbol queries.
* ``TEST_V4_QUERIES`` (30) -- **retired**. Committed before code-first routing and commit-relevance gating; later
  used to diagnose the warning-recall drop and the regression-routing gap.
* ``TEST_V5_QUERIES`` (30) -- the current held-out split (6 per category), committed with a promotions / CSV-export
  fixture extension *before* the direct top-commit warning trigger and the broadened symptom routing existed.
  Regression queries are natural-language bug reports. Report these as the headline numbers.
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
    # ------------------------------------------- behavioral "why": answered by the code itself
    # Two deliberately target a function whose *file* holds a recorded failure on a different function
    # (beh-01 vs tax_cents, beh-07 vs Consumer.HandleOne): warning about it there is a false alarm.
    EvalQuery(
        "beh-01",
        "behavioral",
        "why does format_cents divide the amount by 100",
        {"sym:backend/utils/money.py::format_cents": 2},
    ),
    EvalQuery("beh-02", "behavioral", "why does extract_sku only accept six digit numbers", {SKU: 2}),
    EvalQuery(
        "beh-03",
        "behavioral",
        "why does get_product look in the cache before asking the repository",
        {"sym:backend/catalog/products.py::get_product": 2},
    ),
    EvalQuery(
        "beh-04",
        "behavioral",
        "why does update_display_name reject names longer than 64 characters",
        {"sym:backend/users/profile.py::update_display_name": 2},
    ),
    EvalQuery(
        "beh-05",
        "behavioral",
        "why does GatewayClient.charge send an Idempotency-Key header",
        {"sym:backend/payments/gateway.py::GatewayClient.charge": 2},
        acceptable=("F2-idempotency",),
    ),
    EvalQuery("beh-06", "behavioral", "why does leftpad use str.rjust", {"sym:vendor/leftpad/leftpad.py::leftpad": 2}),
    EvalQuery(
        "beh-07",
        "behavioral",
        "why does NewConsumer take the handler as a function argument",
        {"sym:worker/pkg/queue/consumer.go::NewConsumer": 2},
    ),
    EvalQuery(
        "beh-08",
        "behavioral",
        "why does TTLCache.__init__ create a threading lock",
        {"sym:backend/cache/lru.py::TTLCache.__init__": 2},
        acceptable=("F6-cache-lock",),
    ),
]


TEST_V1_QUERIES: list[EvalQuery] = [
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

C_SHIP_CEIL = "commit:fix(shipping): round parcel weight up"
C_SHIP_WEIGHT = "commit:fix(shipping): reject zero or negative parcel weights"
C_SEC150 = "commit:fix(security): constant-time password comparison"
C_SEC152 = "commit:fix(security): raise PBKDF2 iterations"
C_DLQ = "commit:fix(worker): park poison messages"
C_LRU_FIX = "commit:fix(search): drop the unbounded lru_cache"
SHIP = "sym:backend/shipping/rates.py::shipping_cents_for"
SHIP_WIDGET = "sym:web/src/components/ShippingEstimate.tsx::ShippingEstimate"
VERIFY_PW = "sym:backend/auth/password.py::verify_password"
PW_MODULE = "sym:backend/auth/password.py::<module>"
DEAD_LETTER = "sym:worker/pkg/queue/dlq.go::Consumer.DeadLetter"

TEST_V2_QUERIES: list[EvalQuery] = [
    # ------------------------------------------------------------------ causal "why"
    EvalQuery(
        "v2-why-01",
        "causal",
        "why is the parcel weight rounded up instead of down when pricing shipping",
        {C_SHIP_CEIL: 2, SHIP: 1},
        ("F8-shipping-floor",),
    ),
    EvalQuery(
        "v2-why-02",
        "causal",
        "why does verify_password use hmac.compare_digest instead of ==",
        {C_SEC150: 2, VERIFY_PW: 1},
        ("F9-password-eq",),
    ),
    EvalQuery(
        "v2-why-03",
        "causal",
        "why do poison messages get published to a separate topic",
        {C_DLQ: 2, DEAD_LETTER: 1},
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "v2-why-04",
        "causal",
        "what was the reasoning behind removing the cache decorator from normalize_query",
        {C_LRU_FIX: 2, NORMALIZE: 1},
    ),
    EvalQuery(
        "v2-why-05",
        "causal",
        "why does the receipt template carry a Content-Security-Policy meta tag",
        {C_SEC131: 2, "file:backend/templates/receipt.html": 1},
    ),
    EvalQuery(
        "v2-why-06",
        "causal",
        "rationale for using 600k PBKDF2 iterations",
        {C_SEC152: 2, PW_MODULE: 1},
        acceptable=("F9-password-eq",),
    ),
    # ------------------------------------------------------------- bug / regression
    EvalQuery(
        "v2-bug-01",
        "regression",
        "a 150 gram parcel was charged as if it weighed 100 grams",
        {C_SHIP_CEIL: 2, SHIP: 1},
        ("F8-shipping-floor",),
    ),
    EvalQuery(
        "v2-bug-02",
        "regression",
        "search workers ran out of memory after a spike of unique queries",
        {C_LRU_FIX: 2, NORMALIZE: 1},
    ),
    EvalQuery(
        "v2-bug-03",
        "regression",
        "one malformed order message was redelivered forever and blocked the queue",
        {C_DLQ: 2, DEAD_LETTER: 1},
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "v2-bug-04",
        "regression",
        "attackers could learn about stored hashes by measuring login response times",
        {C_SEC150: 2, VERIFY_PW: 1},
        ("F9-password-eq",),
    ),
    EvalQuery(
        "v2-bug-05",
        "regression",
        "clients whose clock runs fast get kicked out of the app straight away",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    EvalQuery(
        "v2-bug-06",
        "regression",
        "concurrent checkouts from two tabs charged an outdated basket",
        {C_STALE: 2, CHECKOUT: 1},
        acceptable=("F3-asyncio",),
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "v2-cpl-01",
        "coupled",
        "add an apac zone to the shipping rate tables",
        {"file:backend/shipping/rates.py": 2, "file:web/src/components/ShippingEstimate.tsx": 2},
        coupled="web/src/components/ShippingEstimate.tsx",
        acceptable=("F8-shipping-floor",),
    ),
    EvalQuery(
        "v2-cpl-02",
        "coupled",
        "change the zone dropdown in the ShippingEstimate component",
        {SHIP_WIDGET: 2, "file:backend/shipping/rates.py": 2},
        coupled="backend/shipping/rates.py",
        acceptable=("F8-shipping-floor",),
    ),
    EvalQuery(
        "v2-cpl-03",
        "coupled",
        "rename the dead_letter_topic key in the worker config",
        {"file:worker/config/worker.yaml": 2, "file:worker/pkg/queue/dlq.go": 2},
        coupled="worker/pkg/queue/dlq.go",
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "v2-cpl-04",
        "coupled",
        "change the PAYMENT_GATEWAY_URL default in settings.py",
        {"file:backend/config/settings.py": 2, "file:deploy/helm/values.yaml": 2},
        coupled="deploy/helm/values.yaml",
        acceptable=("F2-idempotency",),
    ),
    EvalQuery(
        "v2-cpl-05",
        "coupled",
        "make the weight bracket size used by shipping_cents_for configurable",
        {SHIP: 2, "file:web/src/components/ShippingEstimate.tsx": 1},
        coupled="web/src/components/ShippingEstimate.tsx",
        acceptable=("F8-shipping-floor",),
    ),
    EvalQuery(
        "v2-cpl-06",
        "coupled",
        "add a parcel weight column to the orders table",
        {"file:backend/db/schema.sql": 2, ORDER_ROW: 2},
        coupled="backend/db/models.py",
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery(
        "v2-err-01",
        "error-code",
        "ERR_SHIPPING_WEIGHT",
        {SHIP: 2, C_SHIP_WEIGHT: 2, SHIP_WIDGET: 1},
        acceptable=("F8-shipping-floor",),
    ),
    EvalQuery("v2-err-02", "error-code", "INC-2718", {C_SHIP_CEIL: 2, SHIP: 1}, acceptable=("F8-shipping-floor",)),
    EvalQuery("v2-err-03", "error-code", "SEC-150", {C_SEC150: 2, VERIFY_PW: 1}, acceptable=("F9-password-eq",)),
    EvalQuery("v2-err-04", "error-code", "INC-2741 dead letter", {C_DLQ: 2, DEAD_LETTER: 1}),
    EvalQuery("v2-err-05", "error-code", "MemoryError in normalize_query", {NORMALIZE: 2, C_LRU_FIX: 1}),
    EvalQuery(
        "v2-err-06",
        "error-code",
        "assert 534 == 569 in test_150g_parcel_billed_as_200g",
        {SHIP: 2, C_SHIP_CEIL: 1},
        ("F8-shipping-floor",),
    ),
]

C_RELEASE = "commit:fix(inventory): release abandoned reservations"
C_OVERSELL = "commit:fix(inventory): serialise reserve_stock"
C_TTL10 = "commit:feat(inventory): shorten stock reservations"
C_OOS = "commit:fix(web): show ERR_OUT_OF_STOCK inline"
C_PW_HELPERS = "commit:feat(auth): PBKDF2 password hashing helpers"
RESERVE = "sym:backend/inventory/reservations.py::reserve_stock"
RELEASE = "sym:backend/inventory/reservations.py::release_expired"
OOS_ERR = "sym:backend/inventory/reservations.py::OutOfStockError"
COUNTDOWN = "sym:web/src/hooks/useReservationCountdown.ts::useReservationCountdown"
ADD_TO_CART = "sym:web/src/components/AddToCartButton.tsx::AddToCartButton"
ISSUE_TOKEN = "sym:backend/auth/token.py::issue_token"
SIGN_PAYLOAD = "sym:backend/auth/token.py::sign_payload"
CACHE_SET = "sym:backend/cache/lru.py::TTLCache.set"
HASH_PW = "sym:backend/auth/password.py::hash_password"
RESERVATIONS_FILE = "file:backend/inventory/reservations.py"

# Sibling-symbol queries (why-03, why-04, why-06, err-05) ask about a function whose *file* has a recorded failure on a
# different function: a warning there is a false alarm by construction.
TEST_V3_QUERIES: list[EvalQuery] = [
    # ------------------------------------------------------------------ causal "why"
    EvalQuery(
        "v3-why-01",
        "causal",
        "why does reserve_stock hold a lock around the stock check",
        {C_OVERSELL: 2, RESERVE: 1},
        ("F10-reserve-lock",),
    ),
    EvalQuery(
        "v3-why-02",
        "causal",
        "why is there a scheduled job that frees reservations",
        {C_RELEASE: 2, RELEASE: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v3-why-03", "causal", "why does issue_token sign the payload with an HMAC", {ISSUE_TOKEN: 2, SIGN_PAYLOAD: 2}
    ),
    EvalQuery(
        "v3-why-04", "causal", "why does TTLCache.set evict the oldest entry when the cache is full", {CACHE_SET: 2}
    ),
    EvalQuery(
        "v3-why-05",
        "causal",
        "what is the reason the web countdown uses a ten minute constant",
        {C_TTL10: 2, COUNTDOWN: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery("v3-why-06", "causal", "why does hash_password generate a random salt", {HASH_PW: 2, C_PW_HELPERS: 1}),
    # ------------------------------------------------------------- bug / regression
    EvalQuery(
        "v3-bug-01",
        "regression",
        "two shoppers both bought the last pair of sneakers in stock",
        {C_OVERSELL: 2, RESERVE: 1},
        ("F10-reserve-lock",),
    ),
    EvalQuery(
        "v3-bug-02",
        "regression",
        "items stayed reserved forever when people abandoned checkout",
        {C_RELEASE: 2, RELEASE: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v3-bug-03",
        "regression",
        "the checkout timer said five minutes left but the reservation had expired",
        {COUNTDOWN: 2, RESERVATIONS_FILE: 1},
        ("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v3-bug-04",
        "regression",
        "adding a sold-out item crashed the product page",
        {C_OOS: 2, ADD_TO_CART: 1},
        acceptable=("F10-reserve-lock",),
    ),
    EvalQuery(
        "v3-bug-05",
        "regression",
        "parcels just over 100 grams were undercharged for shipping",
        {C_SHIP_CEIL: 2, SHIP: 1},
        ("F8-shipping-floor",),
    ),
    EvalQuery(
        "v3-bug-06", "regression", "display names with angle brackets broke the receipt mail", {C_SEC131: 2, RENDER: 1}
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "v3-cpl-01",
        "coupled",
        "change how often the reservation release job runs",
        {"file:deploy/cron/jobs.yaml": 2, RELEASE: 2},
        coupled="backend/inventory/reservations.py",
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v3-cpl-02",
        "coupled",
        "extend the stock reservation TTL to 20 minutes",
        {RESERVATIONS_FILE: 2, "file:web/src/hooks/useReservationCountdown.ts": 2},
        ("F11-ttl-mismatch",),
        coupled="web/src/hooks/useReservationCountdown.ts",
    ),
    EvalQuery(
        "v3-cpl-03",
        "coupled",
        "update the duration used by the reservation countdown hook",
        {COUNTDOWN: 2, RESERVATIONS_FILE: 2},
        ("F11-ttl-mismatch",),
        coupled="backend/inventory/reservations.py",
    ),
    EvalQuery(
        "v3-cpl-04",
        "coupled",
        "add a reservation id column to the orders table",
        {"file:backend/db/schema.sql": 2, ORDER_ROW: 2},
        coupled="backend/db/models.py",
    ),
    EvalQuery(
        "v3-cpl-05",
        "coupled",
        "add the uk label to the shipping estimate widget",
        {SHIP_WIDGET: 2, "file:backend/shipping/rates.py": 2},
        coupled="backend/shipping/rates.py",
        acceptable=("F8-shipping-floor",),
    ),
    EvalQuery(
        "v3-cpl-06",
        "coupled",
        "add a new cache key to the helm values",
        {"file:deploy/helm/values.yaml": 2, "file:backend/config/settings.py": 2},
        coupled="backend/config/settings.py",
        acceptable=("F6-cache-lock",),
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery(
        "v3-err-01",
        "error-code",
        "ERR_OUT_OF_STOCK",
        {OOS_ERR: 2, ADD_TO_CART: 2, C_OOS: 1},
        acceptable=("F10-reserve-lock",),
    ),
    EvalQuery("v3-err-02", "error-code", "INC-2811", {C_OVERSELL: 2, RESERVE: 1}, acceptable=("F10-reserve-lock",)),
    EvalQuery("v3-err-03", "error-code", "INC-2803", {C_RELEASE: 2, RELEASE: 1}, acceptable=("F11-ttl-mismatch",)),
    EvalQuery(
        "v3-err-04", "error-code", "release_expired", {RELEASE: 2, C_RELEASE: 1}, acceptable=("F11-ttl-mismatch",)
    ),
    EvalQuery("v3-err-05", "error-code", "sign_payload", {SIGN_PAYLOAD: 2}),
    EvalQuery(
        "v3-err-06",
        "error-code",
        "AssertionError: sold 2 units of sku-000777 but only 1 in stock",
        {RESERVE: 2, C_OVERSELL: 1},
        ("F10-reserve-lock",),
    ),
]

C_MONOTONIC = "commit:fix(api): refill rate-limit buckets from the monotonic clock"
C_LIMIT120 = "commit:feat(api): raise the public rate limit to 120"
C_RETRY_AFTER = "commit:fix(web): honour Retry-After after ERR_RATE_LIMITED"
C_WEBHOOK_REPLAY = "commit:fix(security): refuse webhook signatures older than five minutes"
REFILL = "sym:backend/api/rate_limit.py::TokenBucket.refill"
ALLOW = "sym:backend/api/rate_limit.py::TokenBucket.allow"
BUCKET = "sym:backend/api/rate_limit.py::TokenBucket"
RETRY_AFTER = "sym:backend/api/rate_limit.py::retry_after_seconds"
SIGN_WEBHOOK = "sym:backend/webhooks/signing.py::sign_webhook"
VERIFY_WEBHOOK = "sym:backend/webhooks/signing.py::verify_webhook"
RATE_FILE = "file:backend/api/rate_limit.py"
NGINX_FILE = "file:deploy/nginx/limits.conf"
HTTP_FILE = "file:web/src/api/http.ts"

# Sibling probes (no warning expected): beh-01 retry_after_seconds and err-04 TokenBucket vs the TokenBucket.refill
# failure, beh-03 sign_webhook vs the verify_webhook failure, beh-06 Cart vs the CartService.checkout failure.
TEST_V4_QUERIES: list[EvalQuery] = [
    # -------------------------------------------------------- causal "why" (history)
    EvalQuery(
        "v4-why-01",
        "causal",
        "why was the rate limiter switched to time.monotonic",
        {C_MONOTONIC: 2, REFILL: 1},
        ("F12-wallclock",),
    ),
    EvalQuery(
        "v4-why-02",
        "causal",
        "why are webhook signatures older than five minutes refused",
        {C_WEBHOOK_REPLAY: 2, VERIFY_WEBHOOK: 1},
        ("F13-webhook-replay",),
    ),
    EvalQuery(
        "v4-why-03",
        "causal",
        "why was the public API request limit doubled",
        {C_LIMIT120: 2, RATE_FILE: 1},
        acceptable=("F12-wallclock",),
    ),
    EvalQuery(
        "v4-why-04", "causal", "what led to the web client waiting for Retry-After", {C_RETRY_AFTER: 2, HTTP_FILE: 1}
    ),
    EvalQuery(
        "v4-why-05",
        "causal",
        "why did stock reservations get shortened from 15 to 10 minutes",
        {C_TTL10: 2, RESERVATIONS_FILE: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v4-why-06",
        "causal",
        "why was the lock in TTLCache.get put back",
        {C_REVERT: 2, CACHE_GET: 1},
        ("F6-cache-lock",),
    ),
    # ------------------------------------------- behavioral "why" (answered by code)
    EvalQuery("v4-beh-01", "behavioral", "why does retry_after_seconds round up", {RETRY_AFTER: 2}),
    EvalQuery(
        "v4-beh-02",
        "behavioral",
        "why does TokenBucket.allow call refill before spending tokens",
        {ALLOW: 2, REFILL: 1},
        acceptable=("F12-wallclock",),
    ),
    EvalQuery(
        "v4-beh-03",
        "behavioral",
        "why does sign_webhook include the timestamp in the signed message",
        {SIGN_WEBHOOK: 2},
    ),
    EvalQuery(
        "v4-beh-04",
        "behavioral",
        "why does list_products cut the result down to limit",
        {"sym:backend/catalog/products.py::list_products": 2},
    ),
    EvalQuery("v4-beh-05", "behavioral", "why does render_receipt format the total with two decimals", {RENDER: 2}),
    EvalQuery(
        "v4-beh-06", "behavioral", "why does a new Cart start at version 0", {"sym:backend/cart/service.py::Cart": 2}
    ),
    # ------------------------------------------------------------- bug / regression
    EvalQuery(
        "v4-bug-01",
        "regression",
        "every API client got throttled for forty minutes after the server clock jumped",
        {C_MONOTONIC: 2, REFILL: 1},
        ("F12-wallclock",),
    ),
    EvalQuery(
        "v4-bug-02",
        "regression",
        "an attacker replayed a captured webhook delivery hours later",
        {C_WEBHOOK_REPLAY: 2, VERIFY_WEBHOOK: 1},
        ("F13-webhook-replay",),
    ),
    EvalQuery(
        "v4-bug-03",
        "regression",
        "the browser kept retrying instantly after being rate limited",
        {C_RETRY_AFTER: 2, HTTP_FILE: 1},
    ),
    EvalQuery(
        "v4-bug-04",
        "regression",
        "the last sneaker in stock was reserved by two people at once",
        {C_OVERSELL: 2, RESERVE: 1},
        ("F10-reserve-lock",),
    ),
    EvalQuery(
        "v4-bug-05",
        "regression",
        "products showed as sold out although nobody had bought them",
        {C_RELEASE: 2, RELEASE: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v4-bug-06",
        "regression",
        "tax on some orders came out a cent higher than finance expected",
        {C_BANKER: 2, TAX: 1},
        ("F4-rounding",),
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "v4-cpl-01",
        "coupled",
        "raise the per-client request budget in the rate limiter",
        {RATE_FILE: 2, NGINX_FILE: 2},
        coupled="deploy/nginx/limits.conf",
        acceptable=("F12-wallclock",),
    ),
    EvalQuery(
        "v4-cpl-02",
        "coupled",
        "change the limit_req rate in the nginx config",
        {NGINX_FILE: 2, RATE_FILE: 2},
        coupled="backend/api/rate_limit.py",
        acceptable=("F12-wallclock",),
    ),
    EvalQuery(
        "v4-cpl-03",
        "coupled",
        "make the reservation countdown hook duration configurable",
        {COUNTDOWN: 2, RESERVATIONS_FILE: 2},
        coupled="backend/inventory/reservations.py",
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "v4-cpl-04",
        "coupled",
        "lower base_ms in the worker retry config",
        {"file:worker/config/worker.yaml": 2, BACKOFF: 2},
        coupled="worker/pkg/queue/backoff.go",
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "v4-cpl-05",
        "coupled",
        "rename the subtotal field of the order API response",
        {ORDER_SCHEMA: 2, "file:mobile/src/parsers/orderParser.ts": 2, "file:web/src/api/orderTypes.ts": 1},
        coupled="mobile/src/parsers/orderParser.ts",
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "v4-cpl-06",
        "coupled",
        "add a tracking_url column to the orders table",
        {"file:backend/db/schema.sql": 2, ORDER_ROW: 2},
        coupled="backend/db/models.py",
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery("v4-err-01", "error-code", "ERR_RATE_LIMITED", {C_RETRY_AFTER: 2, RATE_FILE: 1, HTTP_FILE: 1}),
    EvalQuery("v4-err-02", "error-code", "INC-2904", {C_MONOTONIC: 2, REFILL: 1}, acceptable=("F12-wallclock",)),
    EvalQuery(
        "v4-err-03",
        "error-code",
        "SEC-171",
        {C_WEBHOOK_REPLAY: 2, VERIFY_WEBHOOK: 1},
        acceptable=("F13-webhook-replay",),
    ),
    EvalQuery("v4-err-04", "error-code", "TokenBucket", {BUCKET: 2, REFILL: 1}),
    EvalQuery(
        "v4-err-05",
        "error-code",
        "verify_webhook",
        {VERIFY_WEBHOOK: 2, C_WEBHOOK_REPLAY: 1},
        acceptable=("F13-webhook-replay",),
    ),
    EvalQuery(
        "v4-err-06",
        "error-code",
        "AssertionError: bucket empty for 2400s after a -2400s wall-clock step",
        {REFILL: 2, C_MONOTONIC: 1},
        ("F12-wallclock",),
    ),
]

# Dev expansion: natural-language bug reports (silent drops, stale state, boundary cases, "kept doing X"). They may
# reuse facts from retired held-out splits, never their text (see test_no_test_file_contains_held_out_query_text).
DEV_QUERIES += [
    EvalQuery(
        "bug-09",
        "regression",
        "orders were silently dropped whenever a payload could not be parsed",
        {C_DLQ: 2, DEAD_LETTER: 1},
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "bug-10",
        "regression",
        "the request budget stayed empty for ages after the server time was corrected",
        {C_MONOTONIC: 2, REFILL: 1},
        ("F12-wallclock",),
    ),
    EvalQuery(
        "bug-11",
        "regression",
        "stale holds kept stock locked long after shoppers left the site",
        {C_RELEASE: 2, RELEASE: 1},
        acceptable=("F11-ttl-mismatch",),
    ),
    EvalQuery(
        "bug-12",
        "regression",
        "a webhook delivery from yesterday was accepted a second time",
        {C_WEBHOOK_REPLAY: 2, VERIFY_WEBHOOK: 1},
        ("F13-webhook-replay",),
    ),
    EvalQuery(
        "bug-13",
        "regression",
        "parcels slightly over a 100 g step were priced as the lighter step",
        {C_SHIP_CEIL: 2, SHIP: 1},
        ("F8-shipping-floor",),
    ),
    EvalQuery(
        "bug-14",
        "regression",
        "a wrong password got rejected measurably faster than a nearly right one",
        {C_SEC150: 2, VERIFY_PW: 1},
        ("F9-password-eq",),
    ),
    EvalQuery(
        "bug-15",
        "regression",
        "the browser kept hammering the API in a tight loop after a 429",
        {C_RETRY_AFTER: 2, HTTP_FILE: 1},
    ),
    EvalQuery(
        "bug-16",
        "regression",
        "two carts both ended up holding the final unit of an item",
        {C_OVERSELL: 2, RESERVE: 1},
        ("F10-reserve-lock",),
    ),
]

C_COUPON_FLOOR = "commit:fix(promotions): never let stacked coupons push an order below zero"
C_COUPON_EXPIRY = "commit:fix(promotions): keep coupons valid through their whole expiry day"
C_COUPON_UNKNOWN = "commit:feat(promotions): report unknown coupon codes"
C_CSV_QUOTE = "commit:fix(exports): quote CSV cells"
C_CSV_FORMULA = "commit:fix(security): neutralise spreadsheet formulas"
C_BACKOFF100 = "commit:perf(worker): lower the base retry backoff to 100ms"
APPLY_COUPONS = "sym:backend/promotions/coupons.py::apply_coupons"
COUPON_VALID = "sym:backend/promotions/coupons.py::is_coupon_valid"
FIND_COUPON = "sym:backend/promotions/coupons.py::find_coupon"
COUPON_CLASS = "sym:backend/promotions/coupons.py::Coupon"
COUPON_ERROR = "sym:backend/promotions/coupons.py::CouponError"
COUPON_FIELD = "sym:web/src/components/CouponField.tsx::CouponField"
CSV_CELL = "sym:backend/exports/csv_export.py::_cell"
CSV_EXPORT = "sym:backend/exports/csv_export.py::export_orders_csv"
COUPONS_FILE = "file:backend/promotions/coupons.py"
COUPON_FIELD_FILE = "file:web/src/components/CouponField.tsx"

# Sibling probes (no warning expected): beh-01 and err-05 export_orders_csv vs the _cell failure, beh-02
# is_coupon_valid and beh-04 Coupon vs the apply_coupons failure.
TEST_V5_QUERIES: list[EvalQuery] = [
    # -------------------------------------------------------- causal "why" (history)
    EvalQuery(
        "v5-why-01",
        "causal",
        "why was a zero floor added to apply_coupons",
        {C_COUPON_FLOOR: 2, APPLY_COUPONS: 1},
        ("F15-coupon-negative",),
    ),
    EvalQuery(
        "v5-why-02",
        "causal",
        "why did the coupon expiry comparison change from before to on-or-before",
        {C_COUPON_EXPIRY: 2, COUPON_VALID: 1},
    ),
    EvalQuery(
        "v5-why-03",
        "causal",
        "what led to quoting CSV cells that contain commas",
        {C_CSV_QUOTE: 2, CSV_CELL: 1},
        acceptable=("F14-csv-formula",),
    ),
    EvalQuery(
        "v5-why-04",
        "causal",
        "why were spreadsheet formulas escaped in the order export",
        {C_CSV_FORMULA: 2, CSV_CELL: 1},
        ("F14-csv-formula",),
    ),
    EvalQuery(
        "v5-why-05",
        "causal",
        "why was a dedicated error for unknown coupon codes introduced",
        {C_COUPON_UNKNOWN: 2, FIND_COUPON: 1},
    ),
    EvalQuery(
        "v5-why-06",
        "causal",
        "why was the base retry backoff lowered",
        {C_BACKOFF100: 2, BACKOFF: 1},
        acceptable=("F5-ack-order",),
    ),
    # ------------------------------------------- behavioral "why" (answered by code)
    EvalQuery("v5-beh-01", "behavioral", "why does export_orders_csv write a header row first", {CSV_EXPORT: 2}),
    EvalQuery(
        "v5-beh-02", "behavioral", "why does is_coupon_valid accept coupons without an expiry date", {COUPON_VALID: 2}
    ),
    EvalQuery(
        "v5-beh-03",
        "behavioral",
        "why does find_coupon raise CouponError for codes it does not know",
        {FIND_COUPON: 2, COUPON_ERROR: 1},
    ),
    EvalQuery("v5-beh-04", "behavioral", "why does Coupon store percent_off as an integer", {COUPON_CLASS: 2}),
    EvalQuery(
        "v5-beh-05",
        "behavioral",
        "why does reloadCart fetch the cart from the server again",
        {"sym:web/src/api/cartClient.ts::reloadCart": 2},
    ),
    EvalQuery(
        "v5-beh-06", "behavioral", "why does tax_cents use Decimal arithmetic", {TAX: 2}, acceptable=("F4-rounding",)
    ),
    # ------------------------------------------------ natural-language bug reports
    EvalQuery(
        "v5-bug-01",
        "regression",
        "customers ended up with negative order totals after combining coupons",
        {C_COUPON_FLOOR: 2, APPLY_COUPONS: 1},
        ("F15-coupon-negative",),
    ),
    EvalQuery(
        "v5-bug-02",
        "regression",
        "a coupon stopped working on the morning of its last valid day",
        {C_COUPON_EXPIRY: 2, COUPON_VALID: 1},
    ),
    EvalQuery(
        "v5-bug-03",
        "regression",
        "the finance import shifted columns whenever a customer name had a comma",
        {C_CSV_QUOTE: 2, CSV_CELL: 1},
        acceptable=("F14-csv-formula",),
    ),
    EvalQuery(
        "v5-bug-04",
        "regression",
        "opening the export in a spreadsheet ran a formula hidden in a customer name",
        {C_CSV_FORMULA: 2, CSV_CELL: 1},
        ("F14-csv-formula",),
    ),
    EvalQuery(
        "v5-bug-05",
        "regression",
        "mistyped coupon codes only produced a vague error message",
        {C_COUPON_UNKNOWN: 2, COUPON_FIELD: 1},
    ),
    EvalQuery(
        "v5-bug-06",
        "regression",
        "sessions on phones whose clock runs ahead ended right after login",
        {C_SKEW: 2, TOKEN: 1},
        ("F1-clock-skew",),
    ),
    # ------------------------------------------------------------------ coupled files
    EvalQuery(
        "v5-cpl-01",
        "coupled",
        "add a free-shipping coupon type to the promotions module",
        {COUPONS_FILE: 2, COUPON_FIELD_FILE: 2},
        coupled="web/src/components/CouponField.tsx",
        acceptable=("F15-coupon-negative",),
    ),
    EvalQuery(
        "v5-cpl-02",
        "coupled",
        "change the error text the coupon input shows",
        {COUPON_FIELD: 2, COUPONS_FILE: 2},
        coupled="backend/promotions/coupons.py",
    ),
    EvalQuery(
        "v5-cpl-03",
        "coupled",
        "tighten the nginx burst setting for the public API",
        {NGINX_FILE: 2, RATE_FILE: 2},
        coupled="backend/api/rate_limit.py",
        acceptable=("F12-wallclock",),
    ),
    EvalQuery(
        "v5-cpl-04",
        "coupled",
        "add a max_delay key to worker.yaml",
        {"file:worker/config/worker.yaml": 2, BACKOFF: 2},
        coupled="worker/pkg/queue/backoff.go",
        acceptable=("F5-ack-order",),
    ),
    EvalQuery(
        "v5-cpl-05",
        "coupled",
        "add a promo_code field to the order API schema",
        {ORDER_SCHEMA: 2, "file:mobile/src/parsers/orderParser.ts": 2, "file:web/src/api/orderTypes.ts": 1},
        coupled="mobile/src/parsers/orderParser.ts",
        acceptable=("F7-parser-dollars",),
    ),
    EvalQuery(
        "v5-cpl-06",
        "coupled",
        "change the footer of the receipt e-mail template",
        {"file:backend/templates/receipt.html": 2, RENDER: 2},
        coupled="backend/notifications/email.py",
    ),
    # ----------------------------------------------------------- exact identifiers
    EvalQuery("v5-err-01", "error-code", "ERR_COUPON_EXPIRED", {C_COUPON_EXPIRY: 2, COUPON_VALID: 1, COUPON_FIELD: 1}),
    EvalQuery(
        "v5-err-02",
        "error-code",
        "INC-3011",
        {C_COUPON_FLOOR: 2, APPLY_COUPONS: 1},
        acceptable=("F15-coupon-negative",),
    ),
    EvalQuery("v5-err-03", "error-code", "SEC-190", {C_CSV_FORMULA: 2, CSV_CELL: 1}, acceptable=("F14-csv-formula",)),
    EvalQuery("v5-err-04", "error-code", "ERR_COUPON_UNKNOWN", {FIND_COUPON: 2, C_COUPON_UNKNOWN: 2, COUPON_FIELD: 1}),
    EvalQuery("v5-err-05", "error-code", "export_orders_csv", {CSV_EXPORT: 2}),
    EvalQuery(
        "v5-err-06",
        "error-code",
        "assert -1200 >= 0 in test_stacked_coupons_never_go_negative",
        {APPLY_COUPONS: 2, C_COUPON_FLOOR: 1},
        ("F15-coupon-negative",),
    ),
]

TEST_QUERIES = TEST_V5_QUERIES  # current held-out split
QUERIES = DEV_QUERIES  # backwards-compatible alias
SPLITS = {
    "dev": DEV_QUERIES,
    "test_v1": TEST_V1_QUERIES,
    "test_v2": TEST_V2_QUERIES,
    "test_v3": TEST_V3_QUERIES,
    "test_v4": TEST_V4_QUERIES,
    "test_v5": TEST_V5_QUERIES,
}
