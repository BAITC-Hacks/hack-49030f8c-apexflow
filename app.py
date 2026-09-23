"""ApexFlow: an offline Streamlit workspace for AML investigation hypotheses.

Roles, scores and cluster assignments are produced by the pipeline.  This file
only displays those results and the observed source edges.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from ui.data import (
    ROLE_LABELS,
    ViewData,
    ViewDataError,
    file_signature,
    find_node,
    load_view_data,
    parse_gid,
    synthetic_view_data,
)
from ui.graph import build_subgraph, render_svg


DEFAULT_DATA_DIR = os.environ.get("APEXFLOW_DATA_DIR", "/app/data")
DEFAULT_OUTPUT_DIR = os.environ.get("APEXFLOW_OUTPUT_DIR", "/app/output")


@st.cache_data(show_spinner=False)
def _load_cached(data_dir: str, output_dir: str, signature: tuple[tuple[str, int, int], ...]) -> ViewData:
    """Cache only a particular set of file versions, not an arbitrary old run."""

    del signature  # The argument is intentionally part of Streamlit's cache key.
    return load_view_data(data_dir, output_dir)


def _format_score(value: object) -> str:
    return f"{float(value):.3f}"


def _format_kzt(value: object) -> str:
    try:
        return f"{float(value):,.2f}".replace(",", "\u202f") + " KZT"
    except (TypeError, ValueError):
        return "—"


def _bool_label(value: object) -> str:
    return "да" if bool(value) else "нет"


def _top_table(data: ViewData) -> pd.DataFrame:
    clusters = data.nodes_roles[["_gid_key", "cluster_id", "_cluster_key"]]
    result = data.top_nodes.merge(clusters, on="_gid_key", how="left", validate="one_to_one")
    result["role_label"] = result["role"].map(ROLE_LABELS).fillna(result["role"])
    return result


def _filter_top(table: pd.DataFrame, role: str, cluster: str) -> pd.DataFrame:
    filtered = table.copy()
    if role != "Все роли":
        filtered = filtered.loc[filtered["role"] == role]
    if cluster != "Все кластеры":
        filtered = filtered.loc[filtered["_cluster_key"] == cluster]
    return filtered


def _parse_top_gids(raw: object) -> list[str]:
    """Read contract JSON only; never execute text from a CSV cell."""

    try:
        values = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    parsed: list[str] = []
    for value in values:
        try:
            parsed.append(parse_gid(value))
        except ViewDataError:
            continue
    return parsed


def _load_selected_data(use_synthetic: bool) -> ViewData | None:
    if use_synthetic:
        st.warning(
            "Синтетический пример для разработки. Это не результаты анализа исходных данных.",
            icon="⚠️",
        )
        return synthetic_view_data()
    try:
        signature = file_signature(DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR)
        return _load_cached(DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, signature)
    except ViewDataError as error:
        st.error("Результаты расчёта пока недоступны или не проходят проверку данных.")
        st.code(str(error), language=None)
        st.info(
            "После настройки Docker запустите подтверждённый командой pipeline сценарий. "
            "Интерфейс не подменяет отсутствующие результаты синтетическим набором."
        )
        return None


def _set_selected_gid(gid: str) -> None:
    st.session_state["selected_gid"] = gid


def _render_search(data: ViewData) -> None:
    with st.form("gid-search", clear_on_submit=False):
        raw_gid = st.text_input(
            "Поиск по gid",
            placeholder="Введите полный целочисленный идентификатор",
            help="Поиск всегда идёт по полному nodes_roles, а не только по видимому топу.",
        )
        submitted = st.form_submit_button("Найти клиента", type="primary")
    if not submitted:
        return
    try:
        node = find_node(data, raw_gid)
    except ViewDataError as error:
        st.error(str(error))
        return
    if node is None:
        st.warning("Такого gid нет в загруженной выборке.")
        return
    _set_selected_gid(str(node["_gid_key"]))
    st.success(f"Найден gid {node['gid']}.")


def _render_top(data: ViewData, selected_gid: str, role_filter: str, cluster_filter: str) -> None:
    st.subheader("Очередь приоритетной проверки")
    top = _filter_top(_top_table(data), role_filter, cluster_filter)
    st.caption(f"Показано {len(top)} строк глобального топа из {len(data.top_nodes)}.")
    visible = top[["rank", "gid", "role_label", "priority_score", "cluster_id", "why"]].copy()
    visible = visible.rename(
        columns={
            "rank": "Ранг",
            "gid": "gid",
            "role_label": "Роль",
            "priority_score": "Приоритет",
            "cluster_id": "Кластер",
            "why": "Почему в очереди",
        }
    )
    st.dataframe(visible, hide_index=True, use_container_width=True)
    if top.empty:
        st.info("По текущему фильтру нет строк. Поиск по полному набору всё равно доступен.")
        return

    by_gid = {str(row["_gid_key"]): row for _, row in top.iterrows()}
    chosen = st.selectbox(
        "Открыть узел из показанного топа",
        options=list(by_gid),
        format_func=lambda gid: f"#{by_gid[gid]['rank']} · {by_gid[gid]['gid']} · {by_gid[gid]['role_label']}",
        key="top_picker",
    )
    if st.button("Показать выбранный узел", key="open_top"):
        _set_selected_gid(chosen)
        st.rerun()
    if selected_gid not in set(top["_gid_key"].astype(str)):
        st.info(
            "Карточка ниже выбрана поиском по полному набору и не обязана соответствовать текущему фильтру топа."
        )


def _render_node_card(data: ViewData, node: pd.Series) -> None:
    st.subheader("Карточка клиента")
    st.markdown(f"### gid {node['gid']}")
    st.markdown(f"**Роль:** {ROLE_LABELS.get(node['role'], node['role'])} (`{node['role']}`)")
    first, second, third = st.columns(3)
    first.metric("Поддержка гипотезы роли", _format_score(node["role_score"]))
    second.metric("Приоритет проверки", _format_score(node["priority_score"]))
    third.metric("Кластер", str(node["cluster_id"]))
    st.caption(
        "role_score — сила поддержки роли по правилам; priority_score — порядок проверки. "
        "Ни одна из шкал не является вероятностью виновности."
    )
    st.markdown("**Объяснение из расчёта**")
    st.write(str(node["evidence"]))
    st.markdown(
        f"depth: **{node['depth']}** · is_seed: **{_bool_label(node['is_seed'])}**"
    )

    selected_gid = str(node["_gid_key"])
    inbound = data.edges.loc[data.edges["_dst_key"] == selected_gid]
    outbound = data.edges.loc[data.edges["_src_key"] == selected_gid]
    st.markdown(
        "**Наблюдаемые связи:** "
        f"{len(inbound)} входящих на {_format_kzt(inbound['sum_kzt'].sum())}; "
        f"{len(outbound)} исходящих на {_format_kzt(outbound['sum_kzt'].sum())}."
    )

    limitations: list[str] = []
    steps: list[str] = []
    if int(node["depth"]) == 4:
        limitations.append(
            "Выборка обрывается на четвёртом колене; отсутствие исходящих не доказывает, что деньги остались на счёте."
        )
        steps.append("Запросить последующие исходящие операции за пределами глубины текущей выгрузки.")
    if bool(node["is_seed"]):
        limitations.append("Для стартового узла входящие операции до начала выборки могут быть неполными.")
        steps.append("Уточнить входящие операции, предшествующие текущей выборке.")
    if inbound.empty and outbound.empty:
        limitations.append("Наблюдаемых связей нет; это не является ошибкой визуализации.")
    if node["role"] == "transit":
        steps.append("Сопоставить даты доступных входящих и исходящих операций перед выводом о последовательности потоков.")
    if not steps:
        steps.append("Проверить показанные операции и принять решение об углублённой проверке у специалиста.")
    if limitations:
        st.warning("Ограничения наблюдения: " + " ".join(limitations), icon="ℹ️")
    st.markdown("**Следующий шаг аналитика**")
    for step in steps:
        st.write(f"- {step}")


def _render_edge_tables(data: ViewData, selected_gid: str) -> None:
    st.subheader("Наблюдаемые переводы")
    incoming, outgoing = st.columns(2)
    with incoming:
        st.markdown("**Входящие: src → выбранный gid**")
        table = data.edges.loc[data.edges["_dst_key"] == selected_gid, ["src", "sum_kzt", "n_tx"]]
        table = table.sort_values("sum_kzt", ascending=False, kind="stable")
        st.dataframe(table.rename(columns={"src": "src", "sum_kzt": "Сумма, KZT", "n_tx": "Операций"}), hide_index=True, use_container_width=True)
    with outgoing:
        st.markdown("**Исходящие: выбранный gid → dst**")
        table = data.edges.loc[data.edges["_src_key"] == selected_gid, ["dst", "sum_kzt", "n_tx"]]
        table = table.sort_values("sum_kzt", ascending=False, kind="stable")
        st.dataframe(table.rename(columns={"dst": "dst", "sum_kzt": "Сумма, KZT", "n_tx": "Операций"}), hide_index=True, use_container_width=True)


def _render_graph(data: ViewData, selected_gid: str) -> None:
    st.subheader("Направленный граф наблюдаемого окружения")
    left, right, limit_column = st.columns([1.3, 1.3, 1])
    with left:
        graph_mode = st.radio(
            "Область графа",
            options=["neighborhood", "cluster"],
            format_func=lambda value: "Непосредственные соседи" if value == "neighborhood" else "Выбранный кластер",
            horizontal=True,
        )
    with right:
        color_by = st.radio(
            "Окраска",
            options=["role", "cluster"],
            format_func=lambda value: "По ролям" if value == "role" else "По кластерам",
            horizontal=True,
        )
    with limit_column:
        edge_limit = st.slider("Макс. рёбер", min_value=5, max_value=40, value=24, step=1)
    subset = build_subgraph(data.nodes_roles, data.edges, selected_gid, mode=graph_mode, max_edges=edge_limit)
    st.caption(
        f"Показано {len(subset.nodes)} узлов и {len(subset.edges)} наблюдаемых рёбер. "
        + (f"Скрыто по лимиту: {subset.hidden_edges} рёбер." if subset.hidden_edges else "")
    )
    if subset.edges.empty:
        st.info("Наблюдаемых связей нет: показан сам найденный узел.")
    st.components.v1.html(render_svg(subset, selected_gid, color_by=color_by), height=550, scrolling=True)
    if color_by == "role":
        legend = " · ".join(
            f"{label} ({code})" for code, label in ROLE_LABELS.items()
        )
        st.caption("Легенда ролей: " + legend + ". Выбранный узел выделен тёмной рамкой; seed — жёлтой.")
    else:
        st.caption("Каждый кластер окрашен своим стабильным цветом; выбранный узел выделен тёмной рамкой.")


def _render_clusters_and_downloads(data: ViewData) -> None:
    st.subheader("Кластеры и выгрузки")
    cluster_display = data.clusters[
        ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
    ].rename(
        columns={
            "cluster_id": "Кластер",
            "n_nodes": "Узлов",
            "n_seed": "Seed",
            "sum_kzt_internal": "Внутренний оборот, KZT",
            "top_gids": "Топ gid (JSON)",
            "hypothesis": "Гипотеза",
        }
    )
    st.dataframe(cluster_display, hide_index=True, use_container_width=True)
    selected_cluster = st.selectbox(
        "Посмотреть top_gids кластера",
        options=list(data.clusters["_cluster_key"]),
        format_func=lambda key: f"Кластер {data.clusters.loc[data.clusters['_cluster_key'] == key, 'cluster_id'].iloc[0]}",
    )
    cluster = data.clusters.loc[data.clusters["_cluster_key"] == selected_cluster].iloc[0]
    gids = _parse_top_gids(cluster["top_gids"])
    if gids:
        st.caption("Узлы из top_gids: " + ", ".join(gids) + ". Их можно открыть через поиск по gid.")
    else:
        st.warning("top_gids в выбранном кластере не удалось прочитать как JSON-массив целых gid.")

    downloads = st.columns(3)
    for column, filename in zip(downloads, ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"), strict=True):
        with column:
            st.download_button(
                label=f"Скачать {filename}",
                data=data.raw_csv[filename],
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
            )
    st.caption("Скачивание возвращает полный CSV текущего расчёта, а не отфильтрованную таблицу экрана.")


def main() -> None:
    st.set_page_config(page_title="ApexFlow", page_icon="◈", layout="wide")
    st.title("ApexFlow — исследование транзакционной сети")
    st.caption("Рабочее место AML-аналитика: очередь проверки → клиент → связи → гипотеза → ограничения → следующий шаг.")

    with st.sidebar:
        st.header("Данные и фильтры")
        use_synthetic = st.toggle("Синтетический пример для разработки", value=False)
        if st.button("Обновить данные с диска", use_container_width=True):
            _load_cached.clear()
            st.rerun()
    data = _load_selected_data(use_synthetic)
    if data is None:
        st.stop()

    if "selected_gid" not in st.session_state or find_node(data, st.session_state["selected_gid"]) is None:
        _set_selected_gid(str(data.top_nodes.iloc[0]["_gid_key"]))
    selected_gid = str(st.session_state["selected_gid"])

    stats = st.columns(4)
    stats[0].metric("Узлов", len(data.nodes_roles))
    stats[1].metric("Наблюдаемых рёбер", len(data.edges))
    stats[2].metric("Seed", int(data.nodes_roles["is_seed"].astype(bool).sum()))
    stats[3].metric("Кластеров", len(data.clusters))
    st.caption(data.source)

    _render_search(data)
    selected_gid = str(st.session_state["selected_gid"])

    with st.sidebar:
        role_filter = st.selectbox(
            "Роль в таблице топа",
            options=["Все роли", *ROLE_LABELS],
            format_func=lambda value: value if value == "Все роли" else ROLE_LABELS[value],
        )
        cluster_filter = st.selectbox(
            "Кластер в таблице топа",
            options=["Все кластеры", *list(data.clusters["_cluster_key"])],
            format_func=lambda value: value if value == "Все кластеры" else f"Кластер {value}",
        )

    _render_top(data, selected_gid, role_filter, cluster_filter)
    node = find_node(data, selected_gid)
    if node is None:  # Defensive: data was refreshed while a widget retained state.
        st.error("Выбранный gid отсутствует после обновления данных. Выберите его заново.")
        st.stop()

    graph_column, card_column = st.columns([1.45, 1])
    with graph_column:
        _render_graph(data, selected_gid)
    with card_column:
        _render_node_card(data, node)
    _render_edge_tables(data, selected_gid)
    _render_clusters_and_downloads(data)


if __name__ == "__main__":
    main()
