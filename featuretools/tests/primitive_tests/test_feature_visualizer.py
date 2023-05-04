import json
import os
import re

import graphviz
import pytest

from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    FeatureOutputSlice,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature,
    graph_feature,
)
from featuretools.primitives import Count, CumMax, Mode, NMostCommon, Year


@pytest.fixture
def simple_feat(es):
    return IdentityFeature(es["log"].ww["id"])


@pytest.fixture
def trans_feat(es):
    return TransformFeature(IdentityFeature(es["customers"].ww["cancel_date"]), Year)


def test_returns_digraph_object(simple_feat):
    graph = graph_feature(simple_feat)
    assert isinstance(graph, graphviz.Digraph)


def test_saving_png_file(simple_feat, tmpdir):
    output_path = str(tmpdir.join("test1.png"))
    graph_feature(simple_feat, to_file=output_path)
    assert os.path.isfile(output_path)


def test_missing_file_extension(simple_feat):
    output_path = "test1"
    with pytest.raises(ValueError, match="Please use a file extension"):
        graph_feature(simple_feat, to_file=output_path)


def test_invalid_format(simple_feat):
    output_path = "test1.xyz"
    with pytest.raises(ValueError, match="Unknown format"):
        graph_feature(simple_feat, to_file=output_path)


def test_transform(es, trans_feat):
    feat = trans_feat
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = f"0_{feat_name}_year"
    dataframe_table = "\u2605 customers (target)"
    prim_edge = f'customers:cancel_date -> "{prim_node}"'
    feat_edge = f'"{prim_node}" -> customers:"{feat_name}"'

    graph_components = [feat_name, dataframe_table, prim_node, prim_edge, feat_edge]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 3
    to_match = ["customers", "cancel_date", feat_name]
    for match, row in zip(to_match, rows):
        assert match in row


def test_html_symbols(es, tmpdir):
    output_path_template = str(tmpdir.join("test{}.png"))
    value = IdentityFeature(es["log"].ww["value"])
    gt = value > 5
    lt = value < 5
    ge = value >= 5
    le = value <= 5

    for i, feat in enumerate([gt, lt, ge, le]):
        output_path = output_path_template.format(i)
        graph = graph_feature(feat, to_file=output_path).source
        assert os.path.isfile(output_path)
        assert feat.get_name() in graph


def test_groupby_transform(es):
    feat = GroupByTransformFeature(
        IdentityFeature(es["customers"].ww["age"]),
        CumMax,
        IdentityFeature(es["customers"].ww["cohort"]),
    )
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = f"0_{feat_name}_cum_max"
    groupby_node = f"{feat_name}_groupby_customers--cohort"
    dataframe_table = "\u2605 customers (target)"

    groupby_edge = f'customers:cohort -> "{groupby_node}"'
    groupby_input = f'customers:age -> "{groupby_node}"'
    prim_input = f'"{groupby_node}" -> "{prim_node}"'
    feat_edge = f'"{prim_node}" -> customers:"{feat_name}"'

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        dataframe_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    matches = re.findall(r"customers \[label=<\n<TABLE.*?</TABLE>>", graph, re.DOTALL)
    assert len(matches) == 1
    rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
    assert len(rows) == 4
    assert dataframe_table in rows[0]
    assert feat_name in rows[-1]
    assert ("age" in rows[1] and "cohort" in rows[2]) or (
        "age" in rows[2] and "cohort" in rows[1]
    )


def test_groupby_transform_direct_groupby(es):
    groupby = DirectFeature(
        IdentityFeature(es["cohorts"].ww["cohort_name"]), "customers"
    )
    feat = GroupByTransformFeature(
        IdentityFeature(es["customers"].ww["age"]), CumMax, groupby
    )
    graph = graph_feature(feat).source

    groupby_name = groupby.get_name()
    feat_name = feat.get_name()
    join_node = f"1_{groupby_name}_join"
    prim_node = f"0_{feat_name}_cum_max"
    groupby_node = f"{feat_name}_groupby_customers--{groupby_name}"
    customers_table = "\u2605 customers (target)"
    cohorts_table = "cohorts"

    join_groupby = f'"{join_node}" -> customers:cohort'
    join_input = f'cohorts:cohort_name -> "{join_node}"'
    join_out_edge = f'"{join_node}" -> customers:"{groupby_name}"'
    groupby_edge = f'customers:"{groupby_name}" -> "{groupby_node}"'
    groupby_input = f'customers:age -> "{groupby_node}"'
    prim_input = f'"{groupby_node}" -> "{prim_node}"'
    feat_edge = f'"{prim_node}" -> customers:"{feat_name}"'

    graph_components = [
        groupby_name,
        feat_name,
        join_node,
        prim_node,
        groupby_node,
        customers_table,
        cohorts_table,
        join_groupby,
        join_input,
        join_out_edge,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    dataframes = {
        "cohorts": [cohorts_table, "cohort_name"],
        "customers": [customers_table, "cohort", "age", groupby_name, feat_name],
    }
    for dataframe, value in dataframes.items():
        regex = f"{dataframe} \[label=<\n<TABLE.*?</TABLE>>"
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(value)

        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_aggregation(es):
    feat = AggregationFeature(IdentityFeature(es["log"].ww["id"]), "sessions", Count)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = f"0_{feat_name}_count"
    groupby_node = f"{feat_name}_groupby_log--session_id"

    sessions_table = "\u2605 sessions (target)"
    log_table = "log"
    groupby_edge = f'log:session_id -> "{groupby_node}"'
    groupby_input = f'log:id -> "{groupby_node}"'
    prim_input = f'"{groupby_node}" -> "{prim_node}"'
    feat_edge = f'"{prim_node}" -> sessions:"{feat_name}"'

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        sessions_table,
        log_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]

    for component in graph_components:
        assert component in graph

    dataframes = {
        "log": [log_table, "id", "session_id"],
        "sessions": [sessions_table, feat_name],
    }
    for dataframe, value in dataframes.items():
        regex = f"{dataframe} \[label=<\n<TABLE.*?</TABLE>>"
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(value)
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_multioutput(es):
    multioutput = AggregationFeature(
        IdentityFeature(es["log"].ww["zipcode"]), "sessions", NMostCommon
    )
    feat = FeatureOutputSlice(multioutput, 0)
    graph = graph_feature(feat).source

    feat_name = feat.get_name()
    prim_node = f"0_{multioutput.get_name()}_n_most_common"
    groupby_node = f"{multioutput.get_name()}_groupby_log--session_id"

    sessions_table = "\u2605 sessions (target)"
    log_table = "log"
    groupby_edge = f'log:session_id -> "{groupby_node}"'
    groupby_input = f'log:zipcode -> "{groupby_node}"'
    prim_input = f'"{groupby_node}" -> "{prim_node}"'
    feat_edge = f'"{prim_node}" -> sessions:"{feat_name}"'

    graph_components = [
        feat_name,
        prim_node,
        groupby_node,
        sessions_table,
        log_table,
        groupby_edge,
        groupby_input,
        prim_input,
        feat_edge,
    ]

    for component in graph_components:
        assert component in graph

    dataframes = {
        "log": [log_table, "zipcode", "session_id"],
        "sessions": [sessions_table, feat_name],
    }
    for dataframe, value in dataframes.items():
        regex = f"{dataframe} \[label=<\n<TABLE.*?</TABLE>>"
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(value)
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_direct(es):
    d1 = DirectFeature(
        IdentityFeature(es["customers"].ww["engagement_level"]), "sessions"
    )
    d2 = DirectFeature(d1, "log")
    graph = graph_feature(d2).source

    d1_name = d1.get_name()
    d2_name = d2.get_name()
    prim_node1 = f"1_{d1_name}_join"
    prim_node2 = f"0_{d2_name}_join"

    log_table = "\u2605 log (target)"
    sessions_table = "sessions"
    customers_table = "customers"
    groupby_edge1 = f'"{prim_node1}" -> sessions:customer_id'
    groupby_edge2 = f'"{prim_node2}" -> log:session_id'
    groupby_input1 = f'customers:engagement_level -> "{prim_node1}"'
    groupby_input2 = f'sessions:"{d1_name}" -> "{prim_node2}"'
    d1_edge = f'"{prim_node1}" -> sessions:"{d1_name}"'
    d2_edge = f'"{prim_node2}" -> log:"{d2_name}"'

    graph_components = [
        d1_name,
        d2_name,
        prim_node1,
        prim_node2,
        log_table,
        sessions_table,
        customers_table,
        groupby_edge1,
        groupby_edge2,
        groupby_input1,
        groupby_input2,
        d1_edge,
        d2_edge,
    ]
    for component in graph_components:
        assert component in graph

    dataframes = {
        "customers": [customers_table, "engagement_level"],
        "sessions": [sessions_table, "customer_id", d1_name],
        "log": [log_table, "session_id", d2_name],
    }

    for dataframe, value in dataframes.items():
        regex = f"{dataframe} \[label=<\n<TABLE.*?</TABLE>>"
        matches = re.findall(regex, graph, re.DOTALL)
        assert len(matches) == 1

        rows = re.findall(r"<TR.*?</TR>", matches[0], re.DOTALL)
        assert len(rows) == len(value)
        for row in rows:
            matched = False
            for i in dataframes[dataframe]:
                if i in row:
                    matched = True
                    dataframes[dataframe].remove(i)
                    break
            assert matched


def test_stacked(es, trans_feat):
    stacked = AggregationFeature(trans_feat, "cohorts", Mode)
    graph = graph_feature(stacked).source

    feat_name = stacked.get_name()
    intermediate_name = trans_feat.get_name()
    agg_primitive = f"0_{feat_name}_mode"
    trans_primitive = f"1_{intermediate_name}_year"
    groupby_node = f"{feat_name}_groupby_customers--cohort"

    trans_prim_edge = f'customers:cancel_date -> "{trans_primitive}"'
    intermediate_edge = f'"{trans_primitive}" -> customers:"{intermediate_name}"'
    groupby_edge = f'customers:cohort -> "{groupby_node}"'
    groupby_input = f'customers:"{intermediate_name}" -> "{groupby_node}"'
    agg_input = f'"{groupby_node}" -> "{agg_primitive}"'
    feat_edge = f'"{agg_primitive}" -> cohorts:"{feat_name}"'

    graph_components = [
        feat_name,
        intermediate_name,
        agg_primitive,
        trans_primitive,
        groupby_node,
        trans_prim_edge,
        intermediate_edge,
        groupby_edge,
        groupby_input,
        agg_input,
        feat_edge,
    ]
    for component in graph_components:
        assert component in graph

    agg_primitive = agg_primitive.replace("(", "\\(").replace(")", "\\)")
    agg_node = re.findall(f'"{agg_primitive}" \\[label.*', graph)
    assert len(agg_node) == 1
    assert "Step 2" in agg_node[0]

    trans_primitive = trans_primitive.replace("(", "\\(").replace(")", "\\)")
    trans_node = re.findall(f'"{trans_primitive}" \\[label.*', graph)
    assert len(trans_node) == 1
    assert "Step 1" in trans_node[0]


def test_description_auto_caption(trans_feat):
    default_graph = graph_feature(trans_feat, description=True).source
    default_label = 'label="The year of the \\"cancel_date\\"."'
    assert default_label in default_graph


def test_description_auto_caption_metadata(trans_feat, tmpdir):
    feature_descriptions = {"customers: cancel_date": "the date the customer cancelled"}
    primitive_templates = {"year": "the year that {} occurred"}
    metadata_graph = graph_feature(
        trans_feat,
        description=True,
        feature_descriptions=feature_descriptions,
        primitive_templates=primitive_templates,
    ).source

    metadata_label = 'label="The year that the date the customer cancelled occurred."'
    assert metadata_label in metadata_graph

    metadata = {
        "feature_descriptions": feature_descriptions,
        "primitive_templates": primitive_templates,
    }
    metadata_path = os.path.join(tmpdir, "description_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    json_metadata_graph = graph_feature(
        trans_feat, description=True, metadata_file=metadata_path
    ).source
    assert metadata_label in json_metadata_graph


def test_description_custom_caption(trans_feat):
    custom_description = "A custom feature description"
    custom_description_graph = graph_feature(
        trans_feat, description=custom_description
    ).source
    custom_description_label = 'label="A custom feature description"'
    assert custom_description_label in custom_description_graph
