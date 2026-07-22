from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from app.structured_models import (
    StructuredClarification,
    StructuredColumnType,
    StructuredFilter,
    StructuredIntent,
    StructuredUnavailable,
)
from app.structured_query import parse_structured_intent, resolve_structured_intent
from tests.support.structured_fakes import (
    FakeClickHouse,
    RecordingLLMProvider,
    sample_catalog,
    sample_publication,
)


class StructuredIntentParserTest(unittest.TestCase):
    def test_parses_average_with_alias_and_filter(self) -> None:
        intent = parse_structured_intent(
            "统计华东地区订单金额的平均值",
            sample_catalog(),
        )

        self.assertEqual(intent.aggregate, "avg")
        self.assertEqual(intent.metric_physical_name, "order_amount")
        self.assertEqual(intent.filters, (StructuredFilter("region", "eq", "华东"),))

    def test_ambiguous_metric_never_selects_first_column(self) -> None:
        result = resolve_structured_intent("平均金额", sample_catalog(ambiguous=True))

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(set(result.candidates), {"net_amount", "order_amount"})

    def test_independently_mentioned_metrics_clarify_even_when_lengths_differ(self) -> None:
        result = resolve_structured_intent(
            "订单金额和净金额平均值",
            sample_catalog(ambiguous=True),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("net_amount", "order_amount"))

    def test_parses_numeric_and_date_range_filters(self) -> None:
        intent = parse_structured_intent(
            "统计2026-01-01 至 2026-01-31订单金额大于100的总和",
            sample_catalog(),
        )

        self.assertEqual(intent.aggregate, "sum")
        self.assertIn(StructuredFilter("order_amount", "gt", "100"), intent.filters)
        self.assertIn(
            StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),
            intent.filters,
        )

    def test_date_range_without_confirmed_date_field_never_runs_unfiltered(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        without_date = replace(
            catalog,
            datasets=(
                replace(
                    base,
                    schema=replace(
                        base.schema,
                        columns=tuple(
                            column
                            for column in base.schema.columns
                            if column.physical_name != "order_date"
                        ),
                    ),
                ),
            ),
        )

        result = parse_structured_intent(
            "统计2026-01-01 至 2026-01-31订单金额总和",
            without_date,
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_date_range_with_multiple_date_fields_requires_clarification(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        second_date = replace(
            base.schema.columns[2],
            physical_name="delivery_date",
            original_name="配送日期",
            display_name="配送日期",
            aliases=("日期",),
        )
        multiple_dates = replace(
            catalog,
            datasets=(
                replace(
                    base,
                    schema=replace(base.schema, columns=(*base.schema.columns, second_date)),
                ),
            ),
        )

        result = parse_structured_intent(
            "2026-01-01 至 2026-01-31订单金额总和",
            multiple_dates,
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("delivery_date", "order_date"))

    def test_explicit_date_field_is_consumed_before_count_metric_resolution(self) -> None:
        result = parse_structured_intent(
            "订单日期2026-01-01至2026-01-31的订单金额计数",
            sample_catalog(),
        )

        self.assertEqual(result.aggregate, "count")
        self.assertEqual(result.metric_physical_name, "order_amount")
        self.assertEqual(
            result.filters,
            (StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),),
        )

    def test_date_filter_consumes_only_the_bound_date_field_mention(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        catalog = sample_catalog()
        intent = parse_structured_intent(
            "订单日期2026-01-01至2026-01-31的订单日期计数",
            catalog,
        )

        self.assertEqual(intent.aggregate, "count")
        self.assertEqual(intent.metric_physical_name, "order_date")
        self.assertEqual(
            intent.filters,
            (StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),),
        )
        plan = StructuredQueryPlanner(catalog).plan(intent, sample_publication())
        self.assertIn("count(order_date) AS aggregate_value", plan.sql)

    def test_date_field_before_range_is_shared_with_count_metric(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        catalog = sample_catalog()
        intent = parse_structured_intent(
            "订单日期2026-01-01至2026-01-31计数",
            catalog,
        )

        self.assertEqual(intent.metric_physical_name, "order_date")
        self.assertEqual(
            intent.filters,
            (StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),),
        )
        plan = StructuredQueryPlanner(catalog).plan(intent, sample_publication())
        self.assertIn("count(order_date) AS aggregate_value", plan.sql)

    def test_date_field_after_range_is_shared_with_count_metric(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        catalog = sample_catalog()
        intent = parse_structured_intent(
            "2026-01-01至2026-01-31订单日期计数",
            catalog,
        )

        self.assertEqual(intent.metric_physical_name, "order_date")
        self.assertEqual(
            intent.filters,
            (StructuredFilter("order_date", "between", "2026-01-01", "2026-01-31"),),
        )
        plan = StructuredQueryPlanner(catalog).plan(intent, sample_publication())
        self.assertIn("count(order_date) AS aggregate_value", plan.sql)

    def test_multiple_and_date_ranges_are_rejected(self) -> None:
        result = parse_structured_intent(
            "2026-01-01至2026-01-07且2026-02-01至2026-02-07的订单金额总和",
            sample_catalog(),
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_supports_all_governed_aggregate_words(self) -> None:
        expectations = {
            "订单金额均值": "avg",
            "订单金额求和": "sum",
            "订单金额计数": "count",
            "订单金额最高": "max",
            "订单金额最低": "min",
        }

        for question, expected in expectations.items():
            with self.subTest(question=question):
                result = parse_structured_intent(question, sample_catalog())
                self.assertEqual(result.aggregate, expected)

    def test_multiple_distinct_aggregates_require_clarification(self) -> None:
        result = parse_structured_intent(
            "订单金额最大值和最小值",
            sample_catalog(),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("max", "min"))

    def test_repeated_synonyms_for_one_aggregate_do_not_create_ambiguity(self) -> None:
        result = parse_structured_intent(
            "订单金额最大值也就是最高值",
            sample_catalog(),
        )

        self.assertEqual(result.aggregate, "max")

    def test_aggregate_words_inside_metric_name_are_not_count_intents(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        sales_quantity = replace(
            base.schema.columns[0],
            physical_name="sales_quantity",
            original_name="销售数量",
            display_name="销售数量",
            aliases=("销量",),
        )
        catalog = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(sales_quantity,))),),
        )

        average = parse_structured_intent("销售数量平均值", catalog)
        total = parse_structured_intent("销售数量总和", catalog)

        self.assertEqual(
            (average.aggregate, average.metric_physical_name), ("avg", "sales_quantity")
        )
        self.assertEqual((total.aggregate, total.metric_physical_name), ("sum", "sales_quantity"))

    def test_count_suffix_remains_valid_when_quantity_is_inside_metric_name(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        order_quantity = replace(
            base.schema.columns[0],
            physical_name="order_quantity",
            original_name="订单数量",
            display_name="订单数量",
            aliases=("订单数",),
        )
        catalog = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(order_quantity,))),),
        )

        result = parse_structured_intent("订单数量计数", catalog)

        self.assertEqual(
            (result.aggregate, result.metric_physical_name), ("count", "order_quantity")
        )

    def test_prefix_aggregate_word_is_not_count_when_metric_is_quantity(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        quantity = replace(
            base.schema.columns[0],
            physical_name="quantity",
            original_name="数量",
            display_name="数量",
            aliases=("件数",),
        )
        catalog = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(quantity,))),),
        )

        result = parse_structured_intent("平均数量", catalog)

        self.assertEqual((result.aggregate, result.metric_physical_name), ("avg", "quantity"))

    def test_metric_resolution_excludes_consumed_explicit_filter_field(self) -> None:
        result = parse_structured_intent(
            "地区=华东的订单金额计数",
            sample_catalog(),
        )

        self.assertEqual(result.aggregate, "count")
        self.assertEqual(result.metric_physical_name, "order_amount")
        self.assertEqual(result.filters, (StructuredFilter("region", "eq", "华东"),))

    def test_non_count_metric_ignores_distinct_numeric_filter_field(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        quantity = replace(
            base.schema.columns[0],
            physical_name="quantity",
            original_name="数量",
            display_name="数量",
            aliases=("件数",),
        )
        catalog = replace(
            catalog,
            datasets=(
                replace(
                    base,
                    schema=replace(
                        base.schema,
                        columns=(base.schema.columns[0], quantity, *base.schema.columns[1:]),
                    ),
                ),
            ),
        )

        result = parse_structured_intent(
            "数量大于100的订单金额总和",
            catalog,
        )

        self.assertEqual(result.aggregate, "sum")
        self.assertEqual(result.metric_physical_name, "order_amount")
        self.assertIn(StructuredFilter("quantity", "gt", "100"), result.filters)

    def test_non_count_uses_single_aggregate_filter_field_when_no_other_metric_exists(self) -> None:
        result = parse_structured_intent(
            "订单金额大于100的总和",
            sample_catalog(),
        )

        self.assertEqual(result.metric_physical_name, "order_amount")
        self.assertEqual(
            result.filters,
            (StructuredFilter("order_amount", "gt", "100"),),
        )

    def test_count_resolves_confirmed_non_aggregate_field(self) -> None:
        result = parse_structured_intent("地区计数", sample_catalog())

        self.assertEqual(result.aggregate, "count")
        self.assertEqual(result.metric_physical_name, "region")

    def test_resolves_normalized_physical_dataset_and_column_names(self) -> None:
        result = parse_structured_intent(
            "DS SALES的ORDER_AMOUNT平均值",
            sample_catalog(),
        )

        self.assertEqual(result.dataset_id, "ds-sales")
        self.assertEqual(result.metric_physical_name, "order_amount")

    def test_longer_normalized_physical_name_is_not_shadowed_by_suffix(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        short = replace(
            base.schema.columns[0],
            physical_name="amount",
            original_name="通用值",
            display_name="通用值",
            aliases=(),
        )
        long = replace(
            base.schema.columns[0],
            physical_name="net_amount",
            original_name="净值",
            display_name="净值",
            aliases=(),
        )
        overlapping = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(short, long))),),
        )

        result = parse_structured_intent("NET_AMOUNT平均值", overlapping)

        self.assertEqual(result.metric_physical_name, "net_amount")

    def test_parses_explicit_equality_and_all_numeric_comparisons(self) -> None:
        equality = parse_structured_intent(
            "地区为华东的订单金额总和",
            sample_catalog(),
        )
        self.assertIn(StructuredFilter("region", "eq", "华东"), equality.filters)

        for word, operator in (("不少于", "gte"), ("小于", "lt"), ("不超过", "lte")):
            with self.subTest(word=word):
                result = parse_structured_intent(
                    f"订单金额{word}100的总和",
                    sample_catalog(),
                )
                self.assertIn(StructuredFilter("order_amount", operator, "100"), result.filters)

    def test_composite_and_filters_keep_equality_value_bounded(self) -> None:
        result = parse_structured_intent(
            "地区=华东且订单金额大于100的总和",
            sample_catalog(),
        )

        self.assertEqual(
            result.filters,
            (
                StructuredFilter("region", "eq", "华东"),
                StructuredFilter("order_amount", "gt", "100"),
            ),
        )

    def test_or_filters_are_rejected_instead_of_compiled_as_and(self) -> None:
        result = parse_structured_intent(
            "地区=华东或地区=华南的订单金额总和",
            sample_catalog(),
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_or_implicit_filters_are_rejected(self) -> None:
        result = parse_structured_intent(
            "华东地区或华南地区的订单金额总和",
            sample_catalog(),
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_or_date_ranges_are_rejected(self) -> None:
        result = parse_structured_intent(
            "2026-01-01至2026-01-07或2026-02-01至2026-02-07订单金额总和",
            sample_catalog(),
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_unknown_explicit_equality_field_is_never_dropped(self) -> None:
        for question in (
            "省份=浙江的订单金额总和",
            "省份为浙江的订单金额总和",
        ):
            with self.subTest(question=question):
                result = parse_structured_intent(question, sample_catalog())
                self.assertIsInstance(
                    result,
                    (StructuredClarification, StructuredUnavailable),
                )

    def test_aggregate_like_unknown_equality_fields_are_rejected(self) -> None:
        for question in (
            "数量=10的订单金额计数",
            "平均值=10的订单金额计数",
        ):
            with self.subTest(question=question):
                result = parse_structured_intent(question, sample_catalog())
                self.assertIsInstance(
                    result,
                    (StructuredClarification, StructuredUnavailable),
                )

    def test_equality_stops_at_chinese_comma_and_de_boundary(self) -> None:
        comma = parse_structured_intent(
            "地区=华东，订单金额大于100的总和",
            sample_catalog(),
        )
        de_boundary = parse_structured_intent(
            "地区=华东的订单金额总和",
            sample_catalog(),
        )

        self.assertIn(StructuredFilter("region", "eq", "华东"), comma.filters)
        self.assertIn(StructuredFilter("region", "eq", "华东"), de_boundary.filters)

    def test_same_field_multiple_and_conditions_are_preserved(self) -> None:
        result = parse_structured_intent(
            "订单金额大于100且订单金额不超过200的总和",
            sample_catalog(),
        )

        self.assertEqual(
            result.filters,
            (
                StructuredFilter("order_amount", "gt", "100"),
                StructuredFilter("order_amount", "lte", "200"),
            ),
        )

    def test_numeric_comparison_rejects_non_numeric_confirmed_columns(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        boolean_column = replace(
            base.schema.columns[1],
            physical_name="is_active",
            original_name="是否有效",
            display_name="是否有效",
            data_type=StructuredColumnType.BOOLEAN,
            aliases=(),
        )
        cases = (
            (catalog, "地区大于100的订单金额总和"),
            (
                replace(
                    catalog,
                    datasets=(
                        replace(
                            base,
                            schema=replace(
                                base.schema,
                                columns=(base.schema.columns[0], boolean_column),
                            ),
                        ),
                    ),
                ),
                "是否有效大于1的订单金额总和",
            ),
            (catalog, "订单日期大于2026-01-01的订单金额总和"),
            (
                replace(
                    catalog,
                    datasets=(
                        replace(
                            base,
                            schema=replace(
                                base.schema,
                                columns=(
                                    base.schema.columns[0],
                                    base.schema.columns[1],
                                    replace(
                                        base.schema.columns[2],
                                        data_type=StructuredColumnType.DATETIME,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
                "订单日期不超过2026-01-31的订单金额总和",
            ),
        )

        for case_catalog, question in cases:
            with self.subTest(question=question):
                result = parse_structured_intent(question, case_catalog)
                self.assertIsInstance(
                    result,
                    (StructuredClarification, StructuredUnavailable),
                )

    def test_numeric_comparison_requires_complete_number_boundary(self) -> None:
        for malformed in ("100abc", "100-200", "100.2.3"):
            with self.subTest(malformed=malformed):
                result = parse_structured_intent(
                    f"订单金额大于{malformed}的总和",
                    sample_catalog(),
                )
                self.assertIsInstance(
                    result,
                    (StructuredClarification, StructuredUnavailable),
                )

    def test_filter_resolves_normalized_physical_name(self) -> None:
        result = parse_structured_intent(
            "ORDER AMOUNT大于100的总和",
            sample_catalog(),
        )

        self.assertIn(StructuredFilter("order_amount", "gt", "100"), result.filters)

    def test_ambiguous_filter_alias_returns_clarification(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        metric = base.schema.columns[0]
        first_filter = replace(
            base.schema.columns[1],
            physical_name="sales_region",
            original_name="销售片区",
            display_name="销售片区",
            aliases=("区域",),
        )
        second_filter = replace(
            base.schema.columns[1],
            physical_name="delivery_region",
            original_name="配送片区",
            display_name="配送片区",
            aliases=("区域",),
        )
        ambiguous = replace(
            catalog,
            datasets=(
                replace(
                    base,
                    schema=replace(
                        base.schema,
                        columns=(metric, first_filter, second_filter),
                    ),
                ),
            ),
        )

        result = parse_structured_intent("区域=华东的订单金额总和", ambiguous)

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("delivery_region", "sales_region"))

    def test_filter_prefers_longest_alias_without_adding_short_alias_filter(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        metric = base.schema.columns[0]
        short = replace(
            base.schema.columns[1],
            physical_name="generic_region",
            original_name="通用片区",
            display_name="通用片区",
            aliases=("地区",),
        )
        long = replace(
            base.schema.columns[1],
            physical_name="sales_region",
            original_name="销售片区",
            display_name="销售片区",
            aliases=("销售地区",),
        )
        aliased = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(metric, short, long))),),
        )

        result = parse_structured_intent("销售地区=华东的订单金额总和", aliased)

        self.assertEqual(result.filters, (StructuredFilter("sales_region", "eq", "华东"),))

    def test_independent_filter_fields_before_one_operator_require_clarification(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        metric = base.schema.columns[0]
        generic = replace(
            base.schema.columns[1],
            physical_name="generic_region",
            original_name="通用片区",
            display_name="通用片区",
            aliases=("地区",),
        )
        sales = replace(
            base.schema.columns[1],
            physical_name="sales_region",
            original_name="销售片区",
            display_name="销售片区",
            aliases=("销售地区",),
        )
        ambiguous = replace(
            catalog,
            datasets=(
                replace(base, schema=replace(base.schema, columns=(metric, generic, sales))),
            ),
        )

        result = parse_structured_intent(
            "销售地区和地区=华东的订单金额总和",
            ambiguous,
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("generic_region", "sales_region"))

    def test_unconfirmed_and_multiple_datasets_are_not_selected(self) -> None:
        catalog = sample_catalog()
        unconfirmed = replace(
            catalog.datasets[0],
            schema=replace(catalog.datasets[0].schema, dataset_id="ds-draft"),
            source_name="draft.xlsx",
            active_publication=None,
        )
        multi_catalog = replace(catalog, datasets=(*catalog.datasets, unconfirmed))

        unavailable = parse_structured_intent("draft订单金额平均值", multi_catalog)
        ambiguous = parse_structured_intent(
            "sales和draft订单金额平均值",
            multi_catalog,
        )

        self.assertEqual(unavailable.message, "指定数据集尚未确认并发布")
        self.assertIsInstance(ambiguous, StructuredClarification)

    def test_dataset_resolution_prefers_longest_normalized_name(self) -> None:
        catalog = sample_catalog()
        sales = catalog.datasets[0]
        regional_publication = replace(
            sample_publication(),
            publication_id="pub-regional",
            dataset_id="ds-regional-sales",
            physical_table_name="structured_ds_regional_sales_v1",
        )
        regional = replace(
            sales,
            schema=replace(sales.schema, dataset_id="ds-regional-sales"),
            source_name="regional-sales.xlsx",
            active_publication=regional_publication,
        )

        result = parse_structured_intent(
            "regional-sales订单金额平均值",
            replace(catalog, datasets=(sales, regional)),
        )

        self.assertEqual(result.dataset_id, "ds-regional-sales")

    def test_dataset_resolution_clarifies_independently_mentioned_nested_names(self) -> None:
        catalog = sample_catalog()
        sales = catalog.datasets[0]
        regional_publication = replace(
            sample_publication(),
            publication_id="pub-regional",
            dataset_id="ds-regional-sales",
            physical_table_name="structured_ds_regional_sales_v1",
        )
        regional = replace(
            sales,
            schema=replace(sales.schema, dataset_id="ds-regional-sales"),
            source_name="regional-sales.xlsx",
            active_publication=regional_publication,
        )

        result = parse_structured_intent(
            "ds-sales和ds-regional-sales订单金额平均值",
            replace(catalog, datasets=(sales, regional)),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("ds-regional-sales", "ds-sales"))

    def test_dataset_resolution_clarifies_independent_cross_priority_mentions(self) -> None:
        catalog = sample_catalog()
        sales = catalog.datasets[0]
        other_publication = replace(
            sample_publication(),
            publication_id="pub-other",
            dataset_id="ds-other",
            physical_table_name="structured_ds_other_v1",
        )
        other = replace(
            sales,
            schema=replace(
                sales.schema,
                dataset_id="ds-other",
                worksheet_name="华南",
            ),
            source_name="other.xlsx",
            active_publication=other_publication,
        )

        result = parse_structured_intent(
            "sales和华南的订单金额平均值",
            replace(catalog, datasets=(sales, other)),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("ds-other", "ds-sales"))

    def test_dataset_resolution_same_priority_and_length_tie_clarifies(self) -> None:
        catalog = sample_catalog()
        first = catalog.datasets[0]
        second_publication = replace(
            sample_publication(),
            publication_id="pub-sales-2",
            dataset_id="ds-sales-2",
            physical_table_name="structured_ds_sales_2_v1",
        )
        second = replace(
            first,
            schema=replace(first.schema, dataset_id="ds-sales-2"),
            source_name="sales.csv",
            active_publication=second_publication,
        )

        result = parse_structured_intent(
            "sales订单金额平均值",
            replace(catalog, datasets=(first, second)),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("ds-sales", "ds-sales-2"))

    def test_dataset_worksheet_substring_inside_metric_is_not_a_mention(self) -> None:
        catalog = sample_catalog()
        sales = catalog.datasets[0]
        other_publication = replace(
            sample_publication(),
            publication_id="pub-other",
            dataset_id="ds-other",
            physical_table_name="structured_ds_other_v1",
        )
        other = replace(
            sales,
            schema=replace(
                sales.schema,
                dataset_id="ds-other",
                worksheet_name="订单",
            ),
            source_name="other.xlsx",
            active_publication=other_publication,
        )

        result = parse_structured_intent(
            "sales订单金额平均值",
            replace(catalog, datasets=(sales, other)),
        )

        self.assertEqual(result.dataset_id, "ds-sales")

    def test_adjacent_chinese_dataset_name_wins_over_metric_substring(self) -> None:
        catalog = sample_catalog()
        orders = replace(
            catalog.datasets[0],
            schema=replace(catalog.datasets[0].schema, worksheet_name="订单"),
        )
        south_publication = replace(
            sample_publication(),
            publication_id="pub-other",
            dataset_id="ds-other",
            physical_table_name="structured_ds_other_v1",
        )
        south = replace(
            catalog.datasets[0],
            schema=replace(
                catalog.datasets[0].schema,
                dataset_id="ds-other",
                worksheet_name="华南",
            ),
            source_name="other.xlsx",
            active_publication=south_publication,
        )

        result = parse_structured_intent(
            "华南订单金额平均值",
            replace(catalog, datasets=(orders, south)),
        )

        self.assertEqual(result.dataset_id, "ds-other")

    def test_equal_dataset_and_column_spans_fail_closed(self) -> None:
        catalog = sample_catalog()
        region_dataset = replace(
            catalog.datasets[0],
            schema=replace(catalog.datasets[0].schema, worksheet_name="地区"),
        )
        other_publication = replace(
            sample_publication(),
            publication_id="pub-other",
            dataset_id="ds-other",
            physical_table_name="structured_ds_other_v1",
        )
        other = replace(
            catalog.datasets[0],
            schema=replace(
                catalog.datasets[0].schema,
                dataset_id="ds-other",
                worksheet_name="其他",
            ),
            source_name="other.xlsx",
            active_publication=other_publication,
        )

        result = parse_structured_intent(
            "地区计数",
            replace(catalog, datasets=(region_dataset, other)),
        )

        self.assertIsInstance(result, (StructuredClarification, StructuredUnavailable))

    def test_exact_dataset_id_wins_same_span_source_stem(self) -> None:
        catalog = sample_catalog()
        exact_publication = replace(
            sample_publication(),
            publication_id="pub-exact",
            dataset_id="sales",
            physical_table_name="structured_sales_v1",
        )
        exact = replace(
            catalog.datasets[0],
            schema=replace(catalog.datasets[0].schema, dataset_id="sales"),
            source_name="exact.xlsx",
            active_publication=exact_publication,
        )
        stem_publication = replace(
            sample_publication(),
            publication_id="pub-stem",
            dataset_id="ds-stem",
            physical_table_name="structured_ds_stem_v1",
        )
        stem = replace(
            catalog.datasets[0],
            schema=replace(catalog.datasets[0].schema, dataset_id="ds-stem"),
            source_name="sales.xlsx",
            active_publication=stem_publication,
        )

        result = parse_structured_intent(
            "sales订单金额平均值",
            replace(catalog, datasets=(exact, stem)),
        )

        self.assertEqual(result.dataset_id, "sales")

    def test_implicit_filter_field_is_not_stolen_by_dataset_worksheet(self) -> None:
        catalog = sample_catalog()
        other_publication = replace(
            sample_publication(),
            publication_id="pub-other",
            dataset_id="ds-other",
            physical_table_name="structured_ds_other_v1",
        )
        other = replace(
            catalog.datasets[0],
            schema=replace(
                catalog.datasets[0].schema,
                dataset_id="ds-other",
                worksheet_name="地区",
            ),
            source_name="other.xlsx",
            active_publication=other_publication,
        )

        result = parse_structured_intent(
            "统计华东地区订单金额的平均值",
            replace(catalog, datasets=(catalog.datasets[0], other)),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("ds-other", "ds-sales"))

    def test_implicit_filter_value_is_not_stolen_by_dataset_worksheet(self) -> None:
        catalog = sample_catalog()
        east_publication = replace(
            sample_publication(),
            publication_id="pub-east",
            dataset_id="ds-east",
            physical_table_name="structured_ds_east_v1",
        )
        east = replace(
            catalog.datasets[0],
            schema=replace(
                catalog.datasets[0].schema,
                dataset_id="ds-east",
                worksheet_name="华东",
            ),
            source_name="east.xlsx",
            active_publication=east_publication,
        )

        result = parse_structured_intent(
            "华东地区订单金额平均值",
            replace(catalog, datasets=(catalog.datasets[0], east)),
        )

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("ds-east", "ds-sales"))

    def test_display_name_tie_returns_all_candidates(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        first = replace(
            base.schema.columns[0],
            physical_name="gross_revenue",
            original_name="收入",
            display_name="收入",
            aliases=("毛收入",),
        )
        second = replace(
            base.schema.columns[0],
            physical_name="net_revenue",
            original_name="收入",
            display_name="收入",
            aliases=("净收入",),
        )
        tied = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(first, second))),),
        )

        result = parse_structured_intent("收入平均值", tied)

        self.assertIsInstance(result, StructuredClarification)
        self.assertEqual(result.candidates, ("gross_revenue", "net_revenue"))

    def test_longest_alias_wins_before_shorter_alias(self) -> None:
        catalog = sample_catalog()
        base = catalog.datasets[0]
        short = replace(
            base.schema.columns[0],
            physical_name="generic_amount",
            original_name="通用指标",
            display_name="通用指标",
            aliases=("金额",),
        )
        long = replace(
            base.schema.columns[0],
            physical_name="order_amount_v2",
            original_name="订单指标",
            display_name="订单指标",
            aliases=("订单金额",),
        )
        aliased = replace(
            catalog,
            datasets=(replace(base, schema=replace(base.schema, columns=(short, long))),),
        )

        result = parse_structured_intent("订单金额平均值", aliased)

        self.assertEqual(result.metric_physical_name, "order_amount_v2")


class StructuredQueryPlannerTest(unittest.TestCase):
    def test_plan_is_select_only_and_aggregate_whitelisted(self) -> None:
        import sqlglot
        from sqlglot import exp

        from app.structured_query import StructuredQueryPlanner

        plan = StructuredQueryPlanner(sample_catalog()).plan(
            StructuredIntent("ds-sales", "avg", "order_amount", ()),
            sample_publication(),
        )

        parsed = sqlglot.parse_one(plan.sql, read="clickhouse")
        functions = {function.sql_name().lower() for function in parsed.find_all(exp.Func)}
        self.assertEqual(parsed.key, "select")
        self.assertEqual(plan.aggregate, "avg")
        self.assertLessEqual(functions, {"avg", "count"})
        self.assertFalse(tuple(parsed.find_all(exp.Join)))
        self.assertFalse(tuple(parsed.find_all(exp.Subquery)))

    def test_filters_are_parameterized_without_raw_value_interpolation(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        plan = StructuredQueryPlanner(sample_catalog()).plan(
            StructuredIntent(
                "ds-sales",
                "sum",
                "order_amount",
                (
                    StructuredFilter("region", "eq", "华东"),
                    StructuredFilter("order_amount", "gt", "100"),
                ),
            ),
            sample_publication(),
        )

        self.assertNotIn("华东", plan.sql)
        self.assertNotIn("> 100", plan.sql)
        self.assertEqual(plan.parameters["filter_0"], "华东")
        self.assertEqual(str(plan.parameters["filter_1"]), "100")
        self.assertIn("{filter_0:String}", plan.sql)
        self.assertIn("{filter_1:Decimal(38, 9)}", plan.sql)

    def test_datetime_date_range_covers_entire_end_day_with_half_open_bound(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        catalog = sample_catalog()
        base = catalog.datasets[0]
        datetime_column = replace(
            base.schema.columns[2],
            data_type=StructuredColumnType.DATETIME,
        )
        catalog = replace(
            catalog,
            datasets=(
                replace(
                    base,
                    schema=replace(
                        base.schema,
                        columns=(base.schema.columns[0], base.schema.columns[1], datetime_column),
                    ),
                ),
            ),
        )
        plan = StructuredQueryPlanner(catalog).plan(
            StructuredIntent(
                "ds-sales",
                "sum",
                "order_amount",
                (
                    StructuredFilter(
                        "order_date",
                        "between",
                        "2026-01-01",
                        "2026-01-31",
                    ),
                ),
            ),
            sample_publication(),
        )

        self.assertIn("order_date < {filter_0_upper:DateTime64(3)}", plan.sql)
        self.assertEqual(plan.parameters["filter_0_upper"], datetime(2026, 2, 1))

    def test_count_all_rows_and_count_non_null_are_distinct(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        planner = StructuredQueryPlanner(sample_catalog())

        all_rows = planner.plan(
            StructuredIntent("ds-sales", "count", None, ()), sample_publication()
        )
        non_null = planner.plan(
            StructuredIntent("ds-sales", "count", "order_amount", ()), sample_publication()
        )

        self.assertIn("count() AS aggregate_value", all_rows.sql)
        self.assertIn("count(order_amount) AS aggregate_value", non_null.sql)
        self.assertIn("count(order_amount) AS valid_count", non_null.sql)

    def test_count_accepts_any_confirmed_field(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        plan = StructuredQueryPlanner(sample_catalog()).plan(
            StructuredIntent("ds-sales", "count", "region", ()), sample_publication()
        )

        self.assertIn("count(region) AS aggregate_value", plan.sql)
        self.assertIn("count(region) AS valid_count", plan.sql)

    def test_sum_min_and_max_use_only_confirmed_metric(self) -> None:
        from app.structured_query import StructuredQueryPlanner

        planner = StructuredQueryPlanner(sample_catalog())

        for aggregate in ("sum", "min", "max"):
            with self.subTest(aggregate=aggregate):
                plan = planner.plan(
                    StructuredIntent("ds-sales", aggregate, "order_amount", ()),
                    sample_publication(),
                )
                self.assertIn(f"{aggregate}(order_amount) AS aggregate_value", plan.sql)

    def test_unknown_columns_and_untrusted_sql_fragments_are_rejected(self) -> None:
        from app.structured_query import StructuredQueryPlanner, UnsafeStructuredQueryError

        planner = StructuredQueryPlanner(sample_catalog())

        for metric in ("missing", "order_amount FROM x", "(SELECT order_amount)"):
            with self.subTest(metric=metric):
                with self.assertRaises(UnsafeStructuredQueryError):
                    planner.plan(
                        StructuredIntent("ds-sales", "avg", metric, ()),
                        sample_publication(),
                    )

        with self.assertRaises(UnsafeStructuredQueryError):
            planner.plan(
                StructuredIntent(
                    "ds-sales",
                    "avg",
                    "order_amount",
                    (StructuredFilter("missing", "eq", "x"),),
                ),
                sample_publication(),
            )


class StructuredQueryExecutorTest(unittest.TestCase):
    def test_gateway_query_uses_only_read_only_client_and_bounded_settings(self) -> None:
        from app.clickhouse_gateway import ClickHouseGateway

        ingest = FakeClickHouse()
        query = FakeClickHouse(aggregate_rows=[(Decimal("20"), 3, 2, 1)])
        gateway = ClickHouseGateway(
            ingest,
            query_client=query,
            max_execution_time=4,
            max_memory_usage=1024,
            max_result_rows=1,
        )

        result = gateway.query("SELECT count()", {"region": "华东"})

        self.assertEqual(result, [(Decimal("20"), 3, 2, 1)])
        self.assertEqual(ingest.queries, [])
        statement, args, kwargs = query.queries[0]
        self.assertEqual(statement, "SELECT count()")
        self.assertEqual(args, ())
        self.assertEqual(kwargs["parameters"], {"region": "华东"})
        self.assertEqual(
            kwargs["settings"],
            {
                "max_execution_time": 4,
                "max_memory_usage": 1024,
                "max_result_rows": 1,
                "overflow_mode": "break",
                "readonly": 1,
            },
        )

    def test_executor_returns_deterministic_governed_metadata(self) -> None:
        from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner

        catalog = sample_catalog()
        plan = StructuredQueryPlanner(catalog).plan(
            StructuredIntent(
                "ds-sales",
                "avg",
                "order_amount",
                (StructuredFilter("region", "eq", "华东"),),
            ),
            sample_publication(),
        )
        gateway = FakeClickHouse(aggregate_rows=[(Decimal("20.5"), 3, 2, 1)])
        times = iter((10.0, 10.025))
        executor = StructuredQueryExecutor(
            catalog,
            gateway,
            clock=lambda: next(times),
            audit_id_factory=lambda: "audit-fixed",
        )

        result = executor.execute(plan)

        self.assertEqual(result.dataset_id, "ds-sales")
        self.assertEqual(result.schema_version, 1)
        self.assertEqual(result.aggregate, "avg")
        self.assertEqual(result.metric_physical_name, "order_amount")
        self.assertEqual(result.metric_display_name, "订单金额")
        self.assertEqual(result.value, Decimal("20.5"))
        self.assertEqual((result.total_count, result.valid_count, result.null_count), (3, 2, 1))
        self.assertEqual(result.source_name, "sales.xlsx")
        self.assertEqual(result.worksheet_name, "明细")
        self.assertEqual(result.publication_id, "pub-sales-1")
        self.assertEqual(result.filters, (StructuredFilter("region", "eq", "华东"),))
        self.assertAlmostEqual(result.elapsed_ms, 25.0)
        self.assertEqual(result.audit_id, "audit-fixed")
        self.assertEqual(gateway.queries[0][0], plan.sql)
        self.assertEqual(gateway.queries[0][1], (plan.parameters,))

    def test_timeout_returns_unavailable_without_fallback(self) -> None:
        from app.structured_models import StructuredUnavailable
        from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner

        class TimeoutGateway:
            def __init__(self) -> None:
                self.calls = 0

            def query(self, statement, parameters):
                self.calls += 1
                raise TimeoutError("timed out")

        catalog = sample_catalog()
        plan = StructuredQueryPlanner(catalog).plan(
            StructuredIntent("ds-sales", "count", None, ()), sample_publication()
        )
        gateway = TimeoutGateway()

        result = StructuredQueryExecutor(catalog, gateway).execute(plan)

        self.assertIsInstance(result, StructuredUnavailable)
        self.assertEqual(gateway.calls, 1)

    def test_structured_query_path_never_calls_llm(self) -> None:
        from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner

        catalog = sample_catalog()
        llm = RecordingLLMProvider()
        intent = parse_structured_intent("订单金额最大值", catalog)
        plan = StructuredQueryPlanner(catalog).plan(intent, sample_publication())
        result = StructuredQueryExecutor(
            catalog,
            FakeClickHouse(aggregate_rows=[(Decimal("30"), 3, 3, 0)]),
        ).execute(plan)

        self.assertEqual(result.value, Decimal("30"))
        self.assertEqual(llm.generation_calls, 0)

    def test_executor_rejects_forged_join_plan_before_gateway_call(self) -> None:
        from app.structured_models import StructuredUnavailable
        from app.structured_query import StructuredQueryExecutor, StructuredQueryPlanner

        catalog = sample_catalog()
        valid = StructuredQueryPlanner(catalog).plan(
            StructuredIntent("ds-sales", "count", None, ()), sample_publication()
        )
        forged = replace(valid, sql=f"{valid.sql} JOIN secret_table ON 1 = 1")
        gateway = FakeClickHouse(aggregate_rows=[(3, 3, 3, 0)])

        result = StructuredQueryExecutor(catalog, gateway).execute(forged)

        self.assertIsInstance(result, StructuredUnavailable)
        self.assertEqual(gateway.queries, [])


if __name__ == "__main__":
    unittest.main()
