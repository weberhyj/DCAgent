from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .bounded_limits import MAX_ROUTE_METADATA_STRING_LENGTH


def normalize_fact_key(value: str) -> str:
    """Create a stable lookup key without changing the display value."""

    if not isinstance(value, str):
        raise TypeError("fact key must be a string")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


FACT_FIELD_ALIASES = {
    "\u5e74\u9f84": (
        "\u5e74\u9f84",
        "\u5e74\u7eaa",
        "\u5c81\u6570",
        "\u51e0\u5c81",
        "\u591a\u5927",
        "age",
    ),
    "\u6027\u522b": ("\u6027\u522b", "\u7537\u5973"),
    "\u804c\u52a1": (
        "\u804c\u52a1",
        "\u804c\u4f4d",
        "\u5c97\u4f4d",
        "\u62c5\u4efb",
        "title",
    ),
}
# Query-time field vocabulary is intentionally broader than the strict Word
# factual extraction registry above. ``FACT_FIELD_ALIASES`` is a data
# contract: adding an alias there changes which values are persisted as exact
# facts. Retrieval/context selection should also understand common synonyms,
# such as ``位置`` in a question and ``主要活动区域`` in a document, without
# changing that exact-fact contract.
QUERY_FIELD_ALIASES = {
    # Put compound age-question forms before the generic quantity aliases.
    # ``有多少岁``/``多少岁`` are natural ways to ask a person's age, not a
    # request to count rows.  Longest-match field extraction will then consume
    # the whole phrase and leave only the entity (plus an optional time
    # qualifier) for subject matching.
    "年龄": FACT_FIELD_ALIASES["年龄"]
    + ("多少岁", "有多少岁", "今年多少岁", "现在多少岁", "现年多少岁"),
    # Do not carry the strict fact parser's bare ``男女`` alias into
    # retrieval: it occurs in unrelated phrases such as ``男女混合活动``.
    # Explicit ``性别``/``男女性别`` wording remains available, while
    # narrative ``男性``/``女性`` is handled by the guarded semantic terms.
    "性别": ("性别", "男女性别", "是男是女", "男还是女", "男或女"),
    "职务": FACT_FIELD_ALIASES["职务"]
    + ("职业", "身份", "工作内容", "工作职责"),
    "位置": (
        "位置",
        "地理位置",
        "所在位置",
        "所在地",
        "地点",
        "所在地点",
        "地址",
        "出生地",
        "出生地点",
        "办公地址",
        "工作地点",
        "居住地",
        "现居地",
        "活动地点",
        "活动区域",
        "主要活动区域",
        "活动范围",
        "驻地",
        "地理区域",
        "在哪",
        "在哪里",
        "什么地方",
        "何处",
    ),
    "姓名": ("姓名", "名字", "名称", "人员姓名", "人物名称", "对象名称"),
    "日期": (
        "日期",
        "时间",
        "日期时间",
        "发生日期",
        "发生时间",
        "时间点",
        "时刻",
        "什么时候",
        "何时",
        "哪天",
    ),
    "地区": ("地区", "所在地区", "所属地区", "地域", "区域", "城市", "省份", "国家"),
    # Keep mutually exclusive measures separate.  ``销售额`` and ``成本``
    # may appear in the same table, but they are not interchangeable answers.
    "金额": ("金额", "总额", "合计金额", "多少钱", "金额多少"),
    "销售额": ("销售额", "销售金额", "销售额多少"),
    "收入": ("收入", "营收", "收入多少"),
    "成本": ("成本", "成本金额", "成本多少"),
    "费用": ("费用", "支出", "费用金额", "费用多少"),
    "数量": (
        "数量",
        "总数",
        "数目",
        "件数",
        "个数",
        "人数",
        "次数",
        "多少个",
        "有多少",
        "几个",
        "数量多少",
    ),
    "联系方式": (
        "电话",
        "联系电话",
        "手机",
        "手机号",
        "联系方式",
        "邮箱",
        "邮件",
        "怎么联系",
        "如何联系",
        "联系方式是什么",
    ),
    # Keep aggregate variants separate too: a request for an average should
    # not be silently bridged to a maximum/minimum column.
    "温度": ("温度", "气温"),
    # ``气温`` is a common raw-temperature header; it is safe for an average
    # request because the structured route performs the requested aggregate.
    # Keep maximum/minimum families separate so an average query cannot drift
    # to an explicitly exclusive extrema column.
    "平均温度": ("平均温度", "平均气温", "均温"),
    "最高温度": ("最高温度", "最高气温"),
    "最低温度": ("最低温度", "最低气温"),
    "湿度": ("湿度", "相对湿度"),
    "降水量": ("降水量", "降雨量", "降水"),
    "风速": ("风速", "风力", "风速值"),
    "距离": ("距离", "里程", "路程"),
    "评分": ("评分", "得分", "分数"),
}
QUERY_FIELD_EXPANSION_MAX_TERMS = 12
QUERY_FIELD_COMPATIBLE_TERMS = {"平均温度": ("温度", "气温")}
# When a query contains both a noun label and its interrogative form (for
# example ``工作地点在哪里``), the interrogative is the actual requested
# field span.  Prefer it for offsets/subject extraction while still expanding
# the complete canonical vocabulary through ``query_field_terms``.
_QUERY_FIELD_INTERROGATIVE_ALIASES = frozenset(
    {"在哪", "在哪里", "什么地方", "何处"}
)
# Narrative evidence does not always repeat a table header verbatim. These
# terms are used only as a guarded semantic signal after the query field has
# already been identified; they are not persisted as exact Word facts and are
# not appended wholesale to sparse retrieval queries.
QUERY_FIELD_SEMANTIC_TERMS = {
    # ``出生`` alone is intentionally excluded: it also appears in fields
    # such as ``出生地`` and would make a location question look like an age
    # hit.  Keep phrases that express an age-bearing relation instead.
    "年龄": ("岁", "出生于", "生于", "出生日期"),
    # Bare ``男``/``女`` are too broad (``男女混合活动``/``女排比赛``).
    # Narrative gender answers normally use one of these complete forms;
    # explicit ``性别：男/女`` remains covered by the field alias itself.
    "性别": ("男性", "女性", "男生", "女生", "男孩", "女孩"),
    # ``工作`` and ``负责`` occur in ordinary task descriptions and are not
    # reliable evidence for a requested job title.  Keep role-bearing verbs
    # and nouns only.
    "职务": ("担任", "任职", "职业", "职称", "职位"),
    # Avoid bare ``活动``/``所在``: they occur in unrelated prose such as
    # ``参加公益活动`` or ``所在部门`` and would promote the chunk during
    # low-confidence lexical fallback. Keep expressions that carry an actual
    # location relation, while explicit aliases such as ``主要活动区域`` and
    # ``所在地`` remain handled by QUERY_FIELD_ALIASES above.
    "位置": (
        "常在",
        "活动在",
        "位于",
        "地处",
        "居住",
        "驻地",
        "坐落",
        "生活在",
        "出生于",
        "生于",
    ),
    # ``联系`` by itself matches prose such as ``联系客户``.  The concrete
    # channel terms are sufficient for narrative contact answers.
    "联系方式": ("电话", "手机", "邮箱", "邮件"),
    "日期": ("发生于", "日期", "时间"),
    "地区": ("位于", "来自", "地区"),
}
_QUERY_FILTER_ONLY_FIELDS = frozenset({"日期", "地区"})
FACT_ENTITY_ALIASES = ("姓名", "人员", "员工", "人物", "名称")
_CANONICAL_FACT_FIELDS = {
    normalize_fact_key(alias): canonical
    for canonical, aliases in FACT_FIELD_ALIASES.items()
    for alias in aliases
}
_CANONICAL_QUERY_FIELDS = {
    normalize_fact_key(alias): field
    for field, aliases in QUERY_FIELD_ALIASES.items()
    for alias in aliases
}
_FILE_REFERENCE_PATTERN = re.compile(
    r"(?:[A-Za-z]:)?[\w./\\-]{1,160}\.(?:docx?|xlsx?|xlsb?|pdf|csv|txt|md)"
    # Chinese query connectors (``中的``/``里的``) may follow a filename;
    # ASCII filename characters may not.  Without this boundary,
    # ``report.xlsxabc`` is silently truncated to ``report.xlsx`` and can
    # become an incorrect hard source constraint.
    r"(?![A-Za-z0-9_.-])(?:中|内|里的)?"
)
_FILE_REFERENCE_NAME_PATTERN = re.compile(
    r"(?:[A-Za-z]:)?[A-Za-z0-9_\-\u4e00-\u9fff./\\]{1,160}\.(?:docx?|xlsx?|xlsb?|pdf|csv|txt|md)"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
_EXPLICIT_FILE_REFERENCE_PATTERN = re.compile(
    r"(?:上传的|文件名|文件|文档|资料|查询|查看|打开|读取)"
    r"(?:[：:\s]*(?:是|为|叫做|叫)?[：:\s]*)?"
    r"(?P<name>(?:[A-Za-z]:)?[A-Za-z0-9_\-\u4e00-\u9fff./\\]{1,160}\.(?:docx?|xlsx?|xlsb?|pdf|csv|txt|md)"
    r"(?![A-Za-z0-9_.-]))",
    re.IGNORECASE,
)

_FILE_REFERENCE_PLACEHOLDER_STEMS = frozenset({"是", "为", "叫做", "叫", "类型", "格式", "后缀"})
_FILE_REFERENCE_COLLISION_PREFIXES = (
    "介绍",
    "查询",
    "查看",
    "打开",
    "读取",
    "说明",
    "解释",
    "阅读",
    "看",
    "一下",
    "关于",
    "在",
    "于",
    "从",
    "文件",
    "文档",
    "资料",
    "上传的",
)
_FILE_REFERENCE_CONVERSATIONAL_SUFFIXES = ("一下", "我", "的", "中", "内", "里")
_FILE_REFERENCE_TYPE_STEM_RE = re.compile(
    r"^(?:文件)?(?:类型|格式|后缀)(?:是|为|叫做|叫)?$"
)
_FILE_REFERENCE_NAME_MARKER_STEM_RE = re.compile(
    r"^文件名(?:是|为|叫做|叫)$"
)


def canonical_fact_field(value: str) -> str:
    clean_value = _bounded_display(value, "field", 120)
    canonical = _CANONICAL_FACT_FIELDS.get(normalize_fact_key(clean_value))
    if canonical is None:
        raise ValueError(f"unknown fact field: {clean_value}")
    return canonical


def canonical_query_field(value: str) -> str:
    """Return the broad retrieval field represented by ``value``.

    This helper is for query interpretation only. Its vocabulary deliberately
    includes loose document/user synonyms and must not be used to persist a
    :class:`KnowledgeFactModel`.
    """

    clean_value = _bounded_display(value, "field", 120)
    canonical = _CANONICAL_QUERY_FIELDS.get(normalize_fact_key(clean_value))
    if canonical is None:
        raise ValueError(f"unknown query field: {clean_value}")
    return canonical


def _bounded_display(value: str, name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    clean_value = value.strip()
    if not clean_value:
        raise ValueError(f"{name} must not be empty")
    if len(clean_value) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return clean_value


def _required_identifier(value: str) -> str:
    return _bounded_display(value, "identifier", 64)


@dataclass(frozen=True, slots=True)
class KnowledgeFactModel:
    id: str
    source_id: str
    chunk_id: str
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str
    value: str
    confidence: float
    locator: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        id: str,
        source_id: str,
        chunk_id: str,
        entity: str,
        field: str,
        value: str,
        confidence: float,
        locator: Mapping[str, object],
    ) -> KnowledgeFactModel:
        clean_entity = _bounded_display(entity, "entity", 240)
        clean_field = canonical_fact_field(field)
        clean_value = _bounded_display(value, "value", 2000)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return cls(
            id=id,
            source_id=_required_identifier(source_id),
            chunk_id=_required_identifier(chunk_id),
            entity=clean_entity,
            entity_normalized=normalize_fact_key(clean_entity),
            field=clean_field,
            field_normalized=normalize_fact_key(clean_field),
            value=clean_value,
            confidence=float(confidence),
            locator=dict(locator),
        )


@dataclass(frozen=True, slots=True)
class WordFactMatch:
    fact: KnowledgeFactModel
    source_name: str
    classification: str


@dataclass(frozen=True, slots=True)
class WordFactualIntent:
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str


@dataclass(frozen=True, slots=True)
class WordFactClarification:
    message: str
    candidates: tuple[str, ...]
    entity: str | None = None
    target_fields: tuple[str, ...] = ()


WordFactualResolution = WordFactualIntent | WordFactClarification | None


@dataclass(frozen=True, slots=True)
class FieldAliasMatch:
    field: str
    alias: str
    start: int
    end: int

    @property
    def normalized_alias(self) -> str:
        """The comparison form used in normalized question/chunk text."""

        return normalize_fact_key(self.alias)


_POLITE_PREFIXES = ("能否告诉我", "帮我查一下", "请问")
_QUESTION_END_PARTICLES = ("吗", "呢")
_FIELD_LIST_FINAL_TAILS = ("是什么", "是多少")
_EXPLICIT_ENTITY_SEPARATORS = ("以及", "、")
_ORGANIZATION_SUFFIXES = (
    "公司",
    "集团",
    "部门",
    "医院",
    "学校",
    "大学",
    "中心",
    "委员会",
    "研究院",
    "事务所",
    "银行",
    "政府",
    "协会",
    "工厂",
)
_FIELD_LIST_ALIASES = {
    "年龄": ("年龄", "年纪", "岁数"),
    "性别": ("性别", "男女"),
    "职务": ("职务", "职位", "岗位"),
}
_FIELD_QUERY_FORMS = {
    "年龄": (
        "几岁",
        "多大",
        "多少岁",
        "有多少岁",
        "今年多少岁",
        "现在多少岁",
        "现年多少岁",
        "年龄",
        "年龄是什么",
        "年龄是多少",
        "年龄多少",
        "年纪",
        "年纪是什么",
        "年纪是多少",
        "年纪多少",
        "岁数",
        "岁数是什么",
        "岁数是多少",
        "岁数多少",
    ),
    "性别": (
        "性别",
        "性别是什么",
        "性别是男是女",
        "是男是女",
        "男还是女",
        "男女",
        "男女是什么",
    ),
    "职务": (
        "职务",
        "职务是什么",
        "职务是做什么",
        "职位",
        "职位是什么",
        "岗位",
        "岗位是什么",
        "担任什么",
        "担任什么职务",
        "担任什么职位",
        "担任什么岗位",
    ),
}


def normalize_question_with_positions(question: str) -> tuple[str, tuple[int, ...]]:
    """Normalize a question while retaining positions in its display text."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized: list[str] = []
    positions: list[int] = []
    for position, character in enumerate(unicodedata.normalize("NFKC", question).casefold()):
        if character.isspace() or (
            character != "、" and unicodedata.category(character).startswith("P")
        ):
            continue
        normalized.append(character)
        positions.append(position)
    return "".join(normalized), tuple(positions)


def find_longest_field_aliases(
    normalized_question: str,
    *,
    aliases: Mapping[str, Sequence[str]] = FACT_FIELD_ALIASES,
) -> tuple[FieldAliasMatch, ...]:
    """Find non-overlapping longest field aliases in normalized question text.

    ``aliases`` is injectable so strict fact parsing can continue using
    :data:`FACT_FIELD_ALIASES`, while retrieval can use the broader
    :data:`QUERY_FIELD_ALIASES` vocabulary.
    """

    if not isinstance(normalized_question, str):
        raise TypeError("normalized_question must be a string")
    candidates: list[FieldAliasMatch] = []
    for field, field_aliases in aliases.items():
        for alias in field_aliases:
            normalized_alias = normalize_fact_key(alias)
            start = normalized_question.find(normalized_alias)
            while start >= 0:
                # ``age`` must not be discovered inside ``average`` and
                # ``title`` must not be discovered inside an identifier. The
                # normalized question has no whitespace, so ASCII word
                # adjacency is the reliable boundary that remains.
                if normalized_alias.isascii() and normalized_alias.isalnum():
                    previous = normalized_question[start - 1] if start > 0 else ""
                    following_index = start + len(normalized_alias)
                    following = (
                        normalized_question[following_index]
                        if following_index < len(normalized_question)
                        else ""
                    )
                    if _is_ascii_word_character(previous) or _is_ascii_word_character(
                        following
                    ):
                        start = normalized_question.find(normalized_alias, start + 1)
                        continue
                candidates.append(
                    FieldAliasMatch(
                        field=field,
                        alias=alias,
                        start=start,
                        end=start + len(normalized_alias),
                    )
                )
                start = normalized_question.find(normalized_alias, start + 1)

    selected: list[FieldAliasMatch] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-(item.end - item.start), item.start, item.field, item.alias),
    ):
        if any(candidate.start < item.end and item.start < candidate.end for item in selected):
            continue
        selected.append(candidate)
    return tuple(sorted(selected, key=lambda item: (item.start, item.end, item.field)))


def _is_ascii_word_character(value: str) -> bool:
    return bool(value) and value.isascii() and (value.isalnum() or value == "_")


def query_file_reference_terms(question: str) -> tuple[str, ...]:
    """Return normalized uploaded-file references explicitly named in a query."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question).casefold())
    references: list[str] = []
    # Prefer a marker-aware capture. A generic filename regex would otherwise
    # consume leading wording (``请查询蜘蛛侠资料.docx``) as part of the file
    # name and make the explicit source constraint impossible to satisfy.
    explicit_matches: list[str] = [
        match.group("name")
        for match in _EXPLICIT_FILE_REFERENCE_PATTERN.finditer(normalized)
    ]
    matches = explicit_matches or [
        match.group(0) for match in _FILE_REFERENCE_NAME_PATTERN.finditer(normalized)
    ]
    for name in matches:
        raw_name = _raw_file_reference_name(name)
        clean_name = _clean_file_reference_name(name)
        compact = normalize_fact_key(clean_name)
        if compact:
            references.append(compact)
        # A valid basename can start with a conversational-looking word
        # (``介绍报告.docx``/``查询报告.xlsx``). Keep that exact candidate in
        # addition to the cleaned wording so hard source scoping can still
        # match the uploaded basename.  Do not keep a raw candidate when the
        # apparent prefix is followed by a conversational suffix such as
        # ``一下``; that is ordinary query scaffolding (``介绍一下报告``), not
        # a likely filename prefix.
        raw_compact = normalize_fact_key(raw_name)
        if (
            raw_compact
            and raw_compact != compact
            and _looks_like_prefixed_basename(raw_name, clean_name)
        ):
            references.append(raw_compact)
    return tuple(dict.fromkeys(references))


def _raw_file_reference_name(value: str) -> str:
    """Return a filename candidate without stripping question prefixes."""

    clean = value.strip().strip("：:，,;；()（）[]【】")
    clean = re.sub(r"(?:中|内|里的)$", "", clean)
    return re.split(r"[/\\]", clean)[-1].strip().strip("：:，,;；()（）[]【】")


def _clean_file_reference_name(value: str) -> str:
    """Remove query scaffolding accidentally captured before a filename."""

    clean = value.strip().strip("：:，,;；()（）[]【】")
    clean = re.sub(r"(?:中|内|里的)$", "", clean)
    # The source registry stores uploaded basenames. Normalize a Windows or
    # Unix path to that same basename before comparing file scope.
    clean = re.split(r"[/\\]", clean)[-1]
    prefixes = (
        "请介绍一下",
        "请介绍",
        "请说明一下",
        "请说明",
        "请解释一下",
        "请解释",
        "请阅读一下",
        "请阅读",
        "请查看一下",
        "请查看",
        "请打开一下",
        "请打开",
        "请查询一下",
        "请查询",
        "请查一下",
        "请帮我打开",
        "请帮我查看",
        "请帮我读取",
        "请帮我阅读",
        "请帮我查询",
        "请帮我查",
        "请看一下",
        "请看",
        "介绍一下",
        "说明一下",
        "解释一下",
        "查询一下",
        "查看一下",
        "读取一下",
        "打开一下",
        "阅读一下",
        "看一下",
        "查一下",
        "我想查看",
        "我想查询",
        "我想读取",
        "我想打开",
        "我想阅读",
        "我想看",
        "我想了解",
        "能不能查看",
        "能不能查询",
        "可以查看",
        "可以查询",
        "帮我打开",
        "帮我查看",
        "帮我读取",
        "帮我阅读",
        "请查询",
        "请查看",
        "请读取",
        "请问",
        "能告诉我",
        "能否告诉我",
        "告诉我",
        "关于",
        "帮我查询",
        "帮我查",
        "请查询",
        "请查看",
        "请读取",
        "查询",
        "查看",
        "打开",
        "读取",
        "介绍",
        "说明",
        "解释",
        "阅读",
        "看",
        "一下",
        "上传的",
        "文件",
        "文档",
        "资料",
        "文件名",
        "在",
        "于",
        "从",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if not clean.startswith(prefix):
                continue
            remainder = clean[len(prefix) :]
            remainder = re.sub(r"^(?:是|为|叫做|叫)", "", remainder)
            # Do not strip a prefix when it is itself the uploaded basename
            # (for example ``介绍.docx`` or ``查询.xlsx``).  A valid remainder
            # must contain at least one filename character before the final
            # extension; bare ``.docx`` is only the suffix left after an
            # accidental prefix match.
            if re.search(
                r"[^./\\]+\.(?:docx?|xlsx?|xlsb?|pdf|csv|txt|md)$",
                remainder,
                re.IGNORECASE,
            ):
                clean = remainder
                changed = True
                break
    # These phrases describe an extension/type, not an uploaded source.  The
    # generic filename regex can otherwise turn ``文件是 .xlsx`` into the
    # bogus source ``是.xlsx`` or ``文件名为.docx`` into ``为.docx``.
    basename = re.split(r"[/\\]", clean)[-1]
    extension_match = re.fullmatch(
        r"(?P<stem>[^/\\]+)\.(?:docx?|xlsx?|xlsb?|pdf|csv|txt|md)",
        basename,
        re.IGNORECASE,
    )
    if extension_match is None:
        return ""
    stem = extension_match.group("stem").strip()
    normalized_stem = normalize_fact_key(stem)
    if (
        not stem
        or normalized_stem in _FILE_REFERENCE_PLACEHOLDER_STEMS
        or _FILE_REFERENCE_TYPE_STEM_RE.fullmatch(stem)
        or _FILE_REFERENCE_NAME_MARKER_STEM_RE.fullmatch(stem)
    ):
        return ""
    return clean.strip().strip("：:，,;；()（）[]【】")


def _looks_like_prefixed_basename(raw_name: str, clean_name: str) -> bool:
    """Decide whether a cleaned filename also needs its raw exact candidate."""

    raw_basename = _raw_file_reference_name(raw_name)
    clean_basename = _raw_file_reference_name(clean_name)
    raw_compact = normalize_fact_key(raw_basename)
    clean_compact = normalize_fact_key(clean_basename)
    if not raw_compact or not clean_compact or raw_compact == clean_compact:
        return False
    # Never preserve an invalid/placeholder raw candidate.
    if not _clean_file_reference_name(raw_basename):
        return False
    for prefix in sorted(_FILE_REFERENCE_COLLISION_PREFIXES, key=len, reverse=True):
        if not raw_basename.startswith(prefix):
            continue
        remainder = raw_basename[len(prefix) :]
        # ``文件/文档/资料`` are query markers when they directly precede a
        # filename (``请读取文件abc.xlsx``).  Keeping the marker-prefixed raw
        # candidate would make the hard-source selector prefer the fictional
        # basename ``文件abc.xlsx`` over the actual upload ``abc.xlsx``.
        # Action-word prefixes (``介绍报告.xlsx``/``查询报告.xlsx``) remain
        # eligible because those can legitimately be part of an uploaded
        # basename and are covered by the collision fallback tests.
        if prefix in {"文件", "文档", "资料"}:
            return False
        # ``一下`` is a standalone conversational prefix in generic matches
        # such as ``查询一下报告.xlsx``; retaining it as a raw source would
        # create a second candidate that is never a real uploaded basename.
        return not (
            prefix == "一下"
            or prefix.endswith("一下")
            or not remainder
            or remainder.startswith(_FILE_REFERENCE_CONVERSATIONAL_SUFFIXES)
        )
    return False


def find_query_field_aliases(question: str) -> tuple[FieldAliasMatch, ...]:
    """Locate broad query-field aliases while retaining their positions.

    The returned offsets refer to the punctuation/whitespace-normalized
    question, which is the same representation used by
    :func:`normalize_fact_key`.  Longest aliases win, so a query containing
    ``地理位置`` returns one match instead of overlapping ``位置``.  Query
    wording can also contain two non-overlapping synonyms for the same field,
    such as ``工作地点在哪里``.  Retrieval only needs one canonical field
    signal, so retain the longest/highest-priority match per field; otherwise
    the shorter noun (``地点``) would be counted as a second field and could
    distort evidence matching.
    """

    # A filename may itself contain a field word (for example
    # ``文件年龄.docx中的内容``).  Parse aliases against a filename-masked
    # representation so that only words outside the uploaded basename become
    # query fields; masking keeps the remaining normalized offsets stable.
    normalized_question = _mask_query_file_reference_text(question)
    normalized, _positions = normalize_question_with_positions(normalized_question)
    matches = find_longest_field_aliases(normalized, aliases=QUERY_FIELD_ALIASES)
    selected_by_field: dict[str, FieldAliasMatch] = {}
    for match in matches:
        current = selected_by_field.get(match.field)
        if current is None:
            selected_by_field[match.field] = match
            continue
        match_is_interrogative = match.alias in _QUERY_FIELD_INTERROGATIVE_ALIASES
        current_is_interrogative = current.alias in _QUERY_FIELD_INTERROGATIVE_ALIASES
        if (
            match_is_interrogative,
            len(match.alias),
            -match.start,
            match.alias,
        ) > (
            current_is_interrogative,
            len(current.alias),
            -current.start,
            current.alias,
        ):
            selected_by_field[match.field] = match
    return tuple(
        sorted(
            selected_by_field.values(),
            key=lambda item: (item.start, item.end, item.field),
        )
    )


def _mask_query_file_reference_text(question: str) -> str:
    """Blank uploaded-file spans before broad query-field extraction.

    ``query_file_reference_terms`` intentionally strips whitespace and query
    scaffolding while extracting basenames.  For field extraction we only
    need the basename span itself masked; keeping all other text (including a
    field after ``.xlsx``/``.docx``) lets queries such as
    ``报告.xlsx中的销售额`` retain the external ``销售额`` field.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question).casefold())
    if not compact:
        return ""
    chars = list(compact)
    for match in _FILE_REFERENCE_NAME_PATTERN.finditer(compact):
        for index in range(match.start(), match.end()):
            chars[index] = " "
    return "".join(chars)


def query_field_terms(question: str) -> tuple[str, ...]:
    """Return canonical field aliases useful for lexical evidence matching.

    If a question mentions one synonym (for example ``位置``), all configured
    variants of that canonical field are returned in deterministic longest-
    first order.  This lets retrieval/context selection match a document that
    uses ``主要活动区域`` without broadening unrelated fields.
    """

    matches = find_query_field_aliases(question)
    fields = tuple(dict.fromkeys(match.field for match in matches))
    terms: list[str] = []
    for field in fields:
        terms.extend(sorted(QUERY_FIELD_ALIASES[field], key=lambda item: (-len(item), item)))
        terms.extend(QUERY_FIELD_COMPATIBLE_TERMS.get(field, ()))
    return tuple(dict.fromkeys(term for term in terms if term))


def query_primary_fields(question: str) -> tuple[str, ...]:
    """Return answer-target fields, excluding date/region filter labels.

    Natural-language spreadsheet questions often contain both filter fields
    (``时间``/``地区``) and the requested measure (``平均温度``/``销售额``).
    Retrieval guards and context selection should anchor on the latter so a
    timestamp-only row cannot masquerade as evidence for the answer metric.
    """

    fields = tuple(dict.fromkeys(match.field for match in find_query_field_aliases(question)))
    primary = tuple(field for field in fields if field not in _QUERY_FILTER_ONLY_FIELDS)
    return primary or fields


def query_primary_field_terms(question: str) -> tuple[str, ...]:
    fields = query_primary_fields(question)
    terms: list[str] = []
    for field in fields:
        terms.extend(sorted(QUERY_FIELD_ALIASES[field], key=lambda item: (-len(item), item)))
        terms.extend(QUERY_FIELD_COMPATIBLE_TERMS.get(field, ()))
    return tuple(dict.fromkeys(term for term in terms if term))


def query_field_matches(question: str, text: str) -> bool:
    """Return whether ``text`` contains a field compatible with ``question``.

    Query aliases bridge user wording and document headers, but aggregate
    qualifiers remain meaningful: an average-temperature query may use a raw
    ``温度``/``气温`` column, yet must not accept an explicitly ``最高`` or
    ``最低`` temperature column as the same field.
    """

    normalized_text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()
    if not normalized_text:
        return False
    for field in query_primary_fields(question):
        for alias in QUERY_FIELD_ALIASES[field]:
            if normalize_fact_key(alias) in normalized_text:
                return True
        for alias in QUERY_FIELD_COMPATIBLE_TERMS.get(field, ()):
            normalized_alias = normalize_fact_key(alias)
            start = normalized_text.find(normalized_alias)
            while start >= 0:
                if field != "平均温度" or not any(
                    normalized_text[max(0, start - len(prefix)) : start] == prefix
                    for prefix in ("最高", "最低", "最大", "最小")
                ):
                    return True
                start = normalized_text.find(normalized_alias, start + 1)
        for semantic_term in QUERY_FIELD_SEMANTIC_TERMS.get(field, ()):
            if _query_semantic_term_matches(field, semantic_term, normalized_text):
                return True
    return False


def _query_semantic_term_matches(field: str, term: str, text: str) -> bool:
    """Match narrative field signals without promoting common unrelated words."""

    normalized_term = normalize_fact_key(term)
    if not normalized_term:
        return False
    if field == "年龄" and normalized_term == normalize_fact_key("岁"):
        # Require a number/quantity immediately before ``岁``.  This avoids
        # treating prose such as ``岁月流逝`` as an age fact.
        return re.search(r"(?:\d+(?:\.\d+)?|[几多])岁", text) is not None
    if normalized_term in {
        normalize_fact_key("出生于"),
        normalize_fact_key("生于"),
    }:
        # A birth marker followed by a year/date is age evidence.  It is not
        # location evidence: ``出生于1990年`` must not satisfy a location
        # question.  Keep the date grammar deliberately bounded so a nearby
        # sentence cannot accidentally turn into a match.
        date_pattern = re.compile(
            r"(?:19|20)\d{2}(?:年(?:\d{1,2}月)?(?:\d{1,2}日)?)?"
            r"|(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?"
        )
        place_pattern = re.compile(
            r"(?:[\u4e00-\u9fff]{2,}(?:省|市|区|县|国|州)?|[A-Za-z][A-Za-z .'-]{1,})"
        )
        start = 0
        while True:
            marker_start = text.find(normalized_term, start)
            if marker_start < 0:
                break
            suffix = text[marker_start + len(normalized_term) :]
            if field == "年龄" and date_pattern.match(suffix) is not None:
                return True
            if (
                field == "位置"
                and date_pattern.match(suffix) is None
                and place_pattern.match(suffix)
            ):
                return True
            start = marker_start + len(normalized_term)
        return False
    return normalized_term in text


def expand_query_field_text(question: str) -> str:
    """Append a bounded set of field synonyms for sparse retrieval.

    Dense retrieval and reranking should continue to receive the user's
    original wording.  BM25-style sparse retrieval benefits from seeing the
    document vocabulary as well, so this helper creates a short auxiliary
    query such as ``位置 主要活动区域 地理位置 所在地``.  It is deliberately
    bounded to keep broad field vocabularies from overwhelming the subject
    terms in a natural-language question.
    """

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    original = question.strip()
    if not original:
        return ""
    normalized_original = normalize_fact_key(original)
    additions = [
        term
        for term in query_field_terms(original)
        if normalize_fact_key(term) not in normalized_original
    ][:QUERY_FIELD_EXPANSION_MAX_TERMS]
    return " ".join([original, *additions])


def query_overlap_terms(question: str) -> tuple[str, ...]:
    """Build bounded lexical terms for query/document overlap checks.

    Field synonyms are expanded only when the question contains a recognized
    query field. Entity and other question words remain available through
    short CJK n-grams and ASCII tokens, preserving the existing behavior for
    ordinary retrieval while adding a precise field bridge.
    """

    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question).casefold())
    if not normalized:
        return ()

    terms: list[str] = []
    terms.extend(re.findall(r"[a-z0-9_]{2,}", normalized))
    for token in re.findall(r"[\u4e00-\u9fff]+", normalized):
        for size in range(2, min(8, len(token)) + 1):
            terms.extend(token[index : index + size] for index in range(len(token) - size + 1))
    terms.extend(query_field_terms(question))
    stop_terms = {"什么", "如何", "怎么", "请问", "一下", "是否", "多少", "是什么"}
    return tuple(
        dict.fromkeys(
            term
            for term in terms
            if term and normalize_fact_key(term) not in stop_terms
        )
    )


_QUERY_SUBJECT_STOP_TERMS = (
    "是什么",
    "是多少",
    "多少",
    "有哪些",
    "哪个",
    "哪些",
    "如何",
    "怎么",
    "请问",
    "请帮我",
    "帮我查一下",
    "帮我查询",
    "能告诉我",
    "能否告诉我",
    "查询",
    "查一下",
    "告诉",
    "告诉我",
    "介绍",
    "说明",
    "帮我查",
    "请介绍",
    "请说明",
    "介绍一下",
    "一下",
    "这个",
    "那个",
    "这份",
    "那份",
    "文件",
    "文档",
    "文档中",
    "资料",
    "资料中",
    "其中",
    "内容",
    "信息",
    "字段",
    "数据",
    "什么",
    "请",
    "能否",
    "所有",
    "全部",
    "各个",
    "平均",
    # Common temporal qualifiers in age questions; they describe when the
    # value is requested, not a second entity/topic (``张三今年几岁``).
    "今年",
    "现在",
    "当前",
    "目前",
    "如今",
    "现年",
    "汇总",
    "合计",
    "总计",
    "总和",
    "范围",
    # These describe the requested operation rather than the entity/topic.
    # Keeping them out of the subject guard prevents questions such as
    # ``张三的年龄变化趋势`` from requiring a chunk to repeat ``变化趋势``.
    "变化趋势",
    "发展趋势",
    "变化情况",
    "趋势",
    "变化",
    "情况",
    "介绍一下",
    "相关",
    "对应",
    "分别",
    "汇报",
    "说明一下",
    "这段",
    "这段时间",
    "期间",
    "时段",
    "分钟",
    "小时",
)
_GENERIC_QUERY_SUBJECT_TERMS = frozenset(
    {
        "公司",
        "部门",
        "岗位",
        "职责",
        "工作",
        "组织",
        "单位",
        "企业",
        "机构",
        "人员",
        "员工",
    }
)
_COMMON_REGION_TERMS = frozenset(
    {
        "中国",
        "北京",
        "上海",
        "天津",
        "重庆",
        "广州",
        "深圳",
        "杭州",
        "南京",
        "苏州",
        "成都",
        "武汉",
        "西安",
        "郑州",
        "济南",
        "青岛",
        "厦门",
        "福州",
        "合肥",
        "长沙",
        "昆明",
        "大连",
        "沈阳",
        "哈尔滨",
        "华东",
        "华南",
        "华北",
        "华中",
        "东北",
        "西南",
        "西北",
        "北美",
        "欧洲",
        "加拿大",
        "美国",
        "多伦多",
        "纽约",
        "伦敦",
        "东京",
    }
)


def query_subject_terms(question: str) -> tuple[str, ...]:
    """Extract bounded entity/topic terms that accompany a query field.

    This is deliberately conservative and is used as a guard for low-score
    evidence acceptance, not as a general-purpose NLP entity recognizer. Field
    aliases and question scaffolding are removed first, leaving terms such as
    ``蜘蛛侠`` or ``2025``. Callers can then require a subject match whenever
    the question contains an explicit subject, preventing a generic document
    with the same field label from being accepted solely on that label.
    """

    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", question).casefold())
    if not normalized:
        return ()

    # Remove explicit file references before the broad field vocabulary. A
    # filename can itself contain a field word (``年龄统计.xlsx``); removing
    # field aliases first would leave a bogus subject such as ``xlsx中``.
    subject_text = _FILE_REFERENCE_PATTERN.sub(" ", normalized)

    # Remove the broad field vocabulary before splitting the remaining text.
    # Longest-first prevents ``主要活动区域`` from being reduced piecemeal to
    # ``位置``-like fragments.
    field_aliases = sorted(
        {
            normalize_fact_key(alias)
            for aliases in QUERY_FIELD_ALIASES.values()
            for alias in aliases
            if alias
        },
        key=lambda item: (-len(item), item),
    )
    for alias in field_aliases:
        subject_text = subject_text.replace(alias, " ")
    for stop_term in sorted(
        (normalize_fact_key(term) for term in _QUERY_SUBJECT_STOP_TERMS),
        key=lambda item: (-len(item), item),
    ):
        subject_text = subject_text.replace(stop_term, " ")

    # Do not remove one-character connectors from the middle of CJK runs:
    # doing so turns names such as ``李和平`` or ``中山大学`` into fragments.
    # Only trim them at the query boundary, beside a non-CJK token/space, or
    # when they introduce a numeric/ASCII qualifier (``华东在2025年``).
    boundary_particles = "的是在于从到与及"
    subject_text = re.sub(
        rf"^[{boundary_particles}]+(?=[\u4e00-\u9fff])",
        " ",
        subject_text,
    )
    subject_text = re.sub(
        rf"[{boundary_particles}]+$",
        " ",
        subject_text,
    )
    subject_text = re.sub(
        rf"(?<![\u4e00-\u9fff])[{boundary_particles}](?![\u4e00-\u9fff])",
        " ",
        subject_text,
    )
    subject_text = re.sub(
        r"(?<=[\u4e00-\u9fff])[在于从到](?=[0-9a-z])",
        " ",
        subject_text,
    )
    subject_text = _split_query_subject_connectors(subject_text)
    # Removing a field leaves the possessive particle in queries such as
    # ``张三的年龄`` (``张三的 ``).  It is structural in that position, but
    # must not be removed from names such as ``李和平``.  Only strip particles
    # adjacent to a replacement boundary/whitespace.
    subject_text = re.sub(r"(?<![\u4e00-\u9fff])[的之](?=\s|$)", " ", subject_text)
    subject_text = re.sub(r"(?<=\s)[的之](?![\u4e00-\u9fff])", " ", subject_text)
    subject_text = re.sub(r"(?<=[\u4e00-\u9fff])[的之](?=\s|$)", " ", subject_text)
    # A trailing possessive particle can remain after punctuation/stop-word
    # removal.  This final boundary cleanup is intentionally narrow.
    subject_text = re.sub(r"[的之]+$", " ", subject_text)
    subject_text = re.sub(r"[^\w\u4e00-\u9fff]+$", "", subject_text)
    subject_text = re.sub(r"的(?=\s*$)", " ", subject_text)
    subject_text = re.sub(r"(?<=\d)[年月日时分秒](?=\d|\s|$)", " ", subject_text)

    terms: list[str] = []
    # Preserve mixed identifiers such as ``A公司`` and ``Q4型号``. A plain
    # ASCII token regex would otherwise discard the one-character prefix,
    # while removing generic suffixes would leave only ``公司``/``型号``.
    mixed_source_tokens: list[str] = []
    mixed_ascii_prefixes: set[str] = set()
    for token in re.findall(r"[a-z0-9_\u4e00-\u9fff]{2,}", subject_text):
        if not (re.search(r"[a-z]", token) and re.search(r"[\u4e00-\u9fff]", token)):
            continue
        mixed_source_tokens.append(token)
        prefix_match = re.match(r"[a-z0-9_]+", token)
        if prefix_match:
            mixed_ascii_prefixes.add(prefix_match.group(0))
        # A mixed identifier can omit the connective before a region, for
        # example ``A公司北京地址``. Split only after a known organization
        # suffix and only when the tail is a recognized region; otherwise
        # retain the complete mixed identifier as one subject term.
        terms.extend(_split_mixed_subject_region(token))
    # Keep alphabetic identifiers as subjects, but do not treat every
    # two-digit clock/date fragment as a hard subject requirement. Years and
    # other long numeric identifiers are added explicitly below.
    terms.extend(
        token
        for token in re.findall(r"[a-z_][a-z0-9_]{1,}", subject_text)
        if token not in mixed_ascii_prefixes
    )
    # Short date fragments (``01``/``08``) and table row numbers are too common
    # to serve as a subject guard. Keep year/identifier-sized numeric tokens;
    # structured Excel date filtering has its own exact route and is not
    # affected by this conservative lexical fallback.
    terms.extend(re.findall(r"\d{4,}", subject_text))
    for token in re.findall(r"[\u4e00-\u9fff]+", subject_text):
        # A CJK substring such as ``公司北京`` is an artefact of a mixed
        # identifier (``A公司北京``), not an independent subject. Keeping it
        # would make the low-score guard require an impossible extra term.
        if any(token in mixed_token for mixed_token in mixed_source_tokens):
            continue
        if len(token) >= 2:
            terms.extend(_split_concatenated_subject_region(token))

    return tuple(
        dict.fromkeys(
            term
            for term in terms
            if term
            and normalize_fact_key(term) not in _GENERIC_QUERY_SUBJECT_TERMS
        )
    )


def _split_concatenated_subject_region(token: str) -> tuple[str, ...]:
    """Split ``组织后缀+地区`` without splitting ordinary names.

    Chinese queries often omit the connective in ``中山大学北京地址``. Only
    split after a known organization suffix when the remaining tail is a
    recognized region; otherwise retain the whole token as one subject.
    """

    for suffix in sorted(_ORGANIZATION_SUFFIXES, key=len, reverse=True):
        start = token.find(suffix)
        if start < 2 or start + len(suffix) >= len(token):
            continue
        left = token[: start + len(suffix)]
        right = token[start + len(suffix) :]
        if right in _COMMON_REGION_TERMS or right.endswith(("省", "市", "区", "县")):
            return left, right
    return (token,)


def _split_mixed_subject_region(token: str) -> tuple[str, ...]:
    """Split a mixed ASCII/CJK organization followed by a region.

    ``_split_concatenated_subject_region`` intentionally refuses a suffix at
    the beginning of a CJK token, because ``公司北京`` is generic text. A
    mixed token has an explicit ASCII/identifier prefix, however, so
    ``A公司北京`` can safely become ``A公司`` and ``北京``.
    """

    for suffix in sorted(_ORGANIZATION_SUFFIXES, key=len, reverse=True):
        start = token.find(suffix)
        if start <= 0 or start + len(suffix) >= len(token):
            continue
        left = token[: start + len(suffix)]
        right = token[start + len(suffix) :]
        # Possessive wording may be retained inside a mixed run after field
        # aliases are removed (``A公司的北京地址`` -> ``A公司的北京``).
        right = right.removeprefix("的")
        if not right:
            continue
        if not re.search(r"[a-z0-9_]", left):
            continue
        if right in _COMMON_REGION_TERMS or right.endswith(("省", "市", "区", "县")):
            return left, right
    return (token,)


def _split_query_subject_connectors(text: str) -> str:
    """Split topic qualifiers without breaking connectors inside names.

    ``中山大学在北京的地址`` should expose ``中山大学`` and ``北京`` as
    separate guard terms, while ``李和平`` must remain one name.  A connector
    is treated as a separator only when both sides look like bounded topic
    spans; this avoids globally deleting one-character CJK particles.
    """

    if not text:
        return text
    chars = list(text)
    connectors = "在于从到和与及、"
    for index, character in enumerate(text):
        if character not in connectors:
            continue
        left_match = re.search(r"([\u4e00-\u9fff]+)$", text[:index].rstrip())
        right_text = text[index + 1 :].lstrip()
        right_match = re.match(r"([\u4e00-\u9fff]+)", right_text)
        left_length = len(left_match.group(1)) if left_match else 0
        right_length = len(right_match.group(1)) if right_match else 0
        numeric_right = re.match(r"\d{4,}", right_text) is not None
        if character == "、":
            chars[index] = " "
        elif character in "在于从到":
            if left_length >= 2 and (right_length >= 2 or numeric_right):
                chars[index] = " "
        elif left_length >= 2 and right_length >= 2:
            # ``张三和李四`` splits; the one-character sides in ``李和平`` do
            # not, preserving the name as a single subject term.
            chars[index] = " "
    return "".join(chars)


def extract_single_entity(
    question: str,
    field_matches: Sequence[FieldAliasMatch],
) -> str | WordFactClarification:
    """Extract the entity only when the complete question matches the exact grammar."""

    del field_matches
    query_match = _match_exact_factual_query(question)
    if query_match is None:
        return WordFactClarification("请使用精确事实问法", ())
    entity, _fields = query_match
    candidates = _entity_list_candidates(entity)
    if candidates is not None:
        return WordFactClarification("一次只能查询一个实体，请选择", candidates)
    if not entity:
        return WordFactClarification("请指定一个实体", ())
    if len(entity) > MAX_ROUTE_METADATA_STRING_LENGTH:
        return _oversized_entity_clarification(_fields)
    return entity


def resolve_word_factual_intent(question: str) -> WordFactualResolution:
    query_match = _match_exact_factual_query(question)
    if query_match is None:
        return None
    entity, fields = query_match
    if len(entity) > MAX_ROUTE_METADATA_STRING_LENGTH:
        return _oversized_entity_clarification(fields)
    if len(fields) != 1:
        return WordFactClarification(
            "一次只能查询一个事实字段，请选择",
            fields,
            entity=entity or None,
            target_fields=fields,
        )
    candidates = _entity_list_candidates(entity)
    if candidates is not None:
        return WordFactClarification(
            "一次只能查询一个实体，请选择",
            candidates,
            target_fields=fields,
        )
    if not entity:
        return WordFactClarification("请指定一个实体", ())
    field = fields[0]
    return WordFactualIntent(
        entity=entity,
        entity_normalized=normalize_fact_key(entity),
        field=field,
        field_normalized=normalize_fact_key(field),
    )


def _oversized_entity_clarification(fields: tuple[str, ...]) -> WordFactClarification:
    return WordFactClarification(
        f"实体名称过长，请提供不超过 {MAX_ROUTE_METADATA_STRING_LENGTH} 个字符的实体名称。",
        (),
        target_fields=fields,
    )


def _match_exact_factual_query(question: str) -> tuple[str, tuple[str, ...]] | None:
    normalized, _positions = normalize_question_with_positions(question)
    for prefix in sorted(_POLITE_PREFIXES, key=len, reverse=True):
        normalized_prefix = normalize_fact_key(prefix)
        if normalized.startswith(normalized_prefix):
            normalized = normalized[len(normalized_prefix) :]
            break
    while normalized.endswith(_QUESTION_END_PARTICLES):
        normalized = normalized[:-1]

    multi_field = _match_multi_field_query(normalized)
    if multi_field is not None:
        return multi_field

    matches: list[tuple[int, str, str]] = []
    for field, forms in _FIELD_QUERY_FORMS.items():
        for form in forms:
            normalized_form = normalize_fact_key(form)
            if normalized.endswith(normalized_form):
                matches.append((len(normalized_form), field, normalized_form))
    if not matches:
        return None
    _length, field, form = max(matches, key=lambda item: (item[0], item[1]))
    entity = _strip_entity_connector(normalized[: -len(form)])
    return entity, (field,)


def _match_multi_field_query(question: str) -> tuple[str, tuple[str, ...]] | None:
    for tail in _FIELD_LIST_FINAL_TAILS:
        if question.endswith(tail):
            question = question[: -len(tail)]
            break
    aliases = sorted(
        (
            (normalize_fact_key(alias), field)
            for field, field_aliases in _FIELD_LIST_ALIASES.items()
            for alias in field_aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    separators = ("以及", "、", "和")
    for start in range(1, len(question)):
        cursor = start
        fields: list[str] = []
        while cursor < len(question):
            alias_match = next(
                ((alias, field) for alias, field in aliases if question.startswith(alias, cursor)),
                None,
            )
            if alias_match is None:
                break
            alias, field = alias_match
            fields.append(field)
            cursor += len(alias)
            if cursor == len(question):
                unique_fields = tuple(dict.fromkeys(fields))
                if len(unique_fields) > 1:
                    return _strip_entity_connector(question[:start]), unique_fields
                break
            separator = next(
                (item for item in separators if question.startswith(item, cursor)),
                None,
            )
            if separator is None:
                break
            cursor += len(separator)
    return None


def _strip_entity_connector(entity: str) -> str:
    return entity.removesuffix("的")


def _entity_list_candidates(entity: str) -> tuple[str, ...] | None:
    for separator in _EXPLICIT_ENTITY_SEPARATORS:
        if separator in entity:
            candidates = tuple(part for part in entity.split(separator) if part)
            if len(candidates) > 1:
                return candidates
    if "和" not in entity:
        return None
    candidates = tuple(part for part in entity.split("和") if part)
    if len(candidates) > 1 and all(_looks_like_bounded_he_list_member(item) for item in candidates):
        return candidates
    return None


def _looks_like_bounded_he_list_member(value: str) -> bool:
    return (
        2 <= len(value) <= 4
        and all("\u3400" <= item <= "\u9fff" for item in value)
        and not value.endswith(_ORGANIZATION_SUFFIXES)
    )


def fact_value_has_embedded_key_value(value: str, *, field: str) -> bool:
    """Return whether a display value contains another apparent key/value record."""

    if not isinstance(value, str):
        return True
    canonical_field = canonical_fact_field(field)
    aliases = list(FACT_ENTITY_ALIASES)
    aliases.extend(
        alias
        for candidate_field, field_aliases in FACT_FIELD_ALIASES.items()
        if candidate_field != canonical_field
        for alias in field_aliases
    )
    normalized_value = unicodedata.normalize("NFKC", value).casefold()
    for alias in aliases:
        normalized_alias = unicodedata.normalize("NFKC", alias).casefold()
        escaped_alias = re.escape(normalized_alias)
        if normalized_alias.isascii():
            pattern = rf"(?<![A-Za-z0-9_]){escaped_alias}(?![A-Za-z0-9_])\s*[:：]"
        else:
            pattern = rf"{escaped_alias}\s*[:：]"
        if re.search(pattern, normalized_value) is not None:
            return True
    return False


def validate_word_fact_answer(
    intent: WordFactualIntent,
    matches: Sequence[WordFactMatch],
    answer: str,
) -> bool:
    """Accept only the bounded template valid for the supplied exact fact matches."""

    if not isinstance(answer, str):
        return False
    display_answer = unicodedata.normalize("NFKC", answer).casefold()
    for field, aliases in FACT_FIELD_ALIASES.items():
        if field == intent.field:
            continue
        if any(_contains_fact_alias(display_answer, alias) for alias in aliases):
            return False

    selected = [
        match
        for match in matches
        if match.fact.entity_normalized == intent.entity_normalized
        and match.fact.field_normalized == intent.field_normalized
    ]
    if any(
        fact_value_has_embedded_key_value(match.fact.value, field=match.fact.field)
        for match in selected
    ):
        return answer == unsafe_word_fact_answer(intent)
    unique_by_source_value: dict[tuple[str, str], WordFactMatch] = {}
    for match in selected:
        unique_by_source_value.setdefault((match.fact.source_id, match.fact.value), match)
    unique_matches = tuple(unique_by_source_value.values())
    if not unique_matches:
        return answer == missing_word_fact_answer(intent)

    source_ids = {match.fact.source_id for match in unique_matches}
    values = {match.fact.value for match in unique_matches}
    if len(source_ids) != 1:
        return answer == source_ambiguity_word_fact_answer(intent)
    if len(values) != 1:
        return answer == conflicting_word_fact_answer(intent)
    value = next(iter(values))
    return answer == exact_word_fact_answer(intent, value)


def exact_word_fact_answer(intent: WordFactualIntent, value: str) -> str:
    return f"{intent.entity}的{intent.field}是{value}。"


def missing_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"未找到{intent.entity}的{intent.field}。"


def source_ambiguity_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"多个来源包含{intent.entity}的{intent.field}记录，请确认来源。"


def conflicting_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"同一来源中存在多个{intent.field}值，请核对来源数据。"


def unsafe_word_fact_answer(intent: WordFactualIntent) -> str:
    return f"无法安全返回{intent.entity}的{intent.field}，请核对来源数据。"


def _contains_fact_alias(display_text: str, alias: str) -> bool:
    normalized_alias = unicodedata.normalize("NFKC", alias).casefold()
    if normalized_alias.isascii():
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized_alias)}(?![A-Za-z0-9_])"
        return re.search(pattern, display_text) is not None
    return normalized_alias in display_text


class WordFactRepository(Protocol):
    def replace_knowledge_facts(self, source_id: str, facts: Sequence[KnowledgeFactModel]) -> None:
        raise NotImplementedError

    def find_knowledge_facts(
        self,
        intent: WordFactualIntent,
        *,
        permission_tags: Sequence[str] = (),
    ) -> list[WordFactMatch]:
        raise NotImplementedError
