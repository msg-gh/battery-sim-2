"""Streamlit UI — Battery Simulator v2 (Perfect Information Oracle)

Run with: streamlit run ui_streamlit.py
"""

import os
import glob
import calendar
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Fixed simulation parameters ---
INVERTER = 9.6        # kW
EFFICIENCY = 0.98
MIN_SOC_PCT = 20      # %
TD_COSTS = 50.0       # $/MWh (T&D costs / retail adder)

MONTH_ABBRS = [calendar.month_abbr[m] for m in range(1, 13)]

st.set_page_config(
    page_title="ERCOT Battery Sim — Perfect Information Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Password gate ---
def check_password():
    """Returns True if the user entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    pwd = st.text_input("Enter password to access the dashboard", type="password")
    if pwd:
        if pwd == st.secrets["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# Tighten padding for dashboard feel
st.markdown("""<style>
    .block-container {padding-top: 0.8rem; padding-bottom: 0;}
    [data-testid="stMetricValue"] {font-size: 1.5rem !important;}
</style>""", unsafe_allow_html=True)

# --- Discover and load result files (CSV or Parquet) ---
# Prefer full results/ (local CSV), fall back to results_deploy/ (cloud Parquet)
_base = os.path.dirname(__file__)
_full = os.path.join(_base, "results")
_deploy = os.path.join(_base, "results_deploy")
RESULTS_DIR = _full if os.path.isdir(_full) else _deploy

@st.cache_data
def find_result_files():
    """Find all result files (CSV or Parquet) in the results directory."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    files = sorted(
        glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
        + glob.glob(os.path.join(RESULTS_DIR, "*.parquet"))
    )
    return [(os.path.basename(f), f) for f in files]

@st.cache_data
def load_data(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if "day" in df.columns:
            df["day"] = pd.to_datetime(df["day"])
    else:
        df = pd.read_csv(path, parse_dates=["day"])
    # Ensure year/month columns exist
    if "year" not in df.columns:
        df["year"] = df["day"].dt.year
    if "month" not in df.columns:
        df["month"] = df["day"].dt.month
    return df

import re as _re

result_files = find_result_files()
if not result_files:
    st.error("No result files found in results/ directory. Run the simulation first.")
    st.stop()

def _extract_capacity(filename):
    """Extract battery capacity in kWh from filename like oracle_25kwh_2020-2025.csv."""
    m = _re.search(r"(\d+)kwh", filename, _re.IGNORECASE)
    return int(m.group(1)) if m else None

# Build list of (capacity, filepath) for files with recognisable capacity in name
capacity_files = []
for name, path in result_files:
    cap = _extract_capacity(name)
    if cap is not None:
        capacity_files.append((cap, path))
capacity_files.sort(key=lambda x: x[0])

if not capacity_files:
    st.error("No result files with recognisable battery capacity found.")
    st.stop()

# --- Title + battery size selector ---
st.title("ERCOT Battery Sim — Perfect Information Model")

available_capacities = [cap for cap, _ in capacity_files]
capacity_to_path = {cap: path for cap, path in capacity_files}

# Default to 25 kWh if available, otherwise first
default_cap_idx = available_capacities.index(25) if 25 in available_capacities else 0
selected_capacity = st.selectbox(
    "Battery Size",
    available_capacities,
    index=default_cap_idx,
    format_func=lambda x: f"{x} kWh Battery",
    label_visibility="collapsed",
)

CAPACITY = float(selected_capacity)
csv_path = capacity_to_path[selected_capacity]
df_all = load_data(csv_path)

st.caption(
    f"**{INVERTER} kW** inverter | "
    f"**{EFFICIENCY:.0%}** efficiency | **{MIN_SOC_PCT}%** min SOC | "
    f"**${TD_COSTS:.0f}/MWh** T&D adder"
)

# ============================================================
# TABS
# ============================================================
tab_summary, tab_details, tab_daily, tab_methodology = st.tabs(
    ["Summary", "Details", "Sim Output", "Methodology"])

# ============================================================
# TAB 1: Daily Detail
# ============================================================
with tab_daily:
    # --- Filters row: house, year, day nav ---
    available_houses = sorted(df_all["house"].unique().tolist())
    available_years = sorted(df_all["year"].unique().tolist())
    has_multiple_houses = len(available_houses) > 1
    has_multiple_years = len(available_years) > 1

    filter_cols = st.columns([2.5, 1.5, 1, 2, 1, 2.5])

    with filter_cols[0]:
        selected_house = st.selectbox(
            "House", available_houses,
            label_visibility="collapsed",
            key="house_select")

    with filter_cols[1]:
        selected_year = st.selectbox(
            "Year", available_years,
            label_visibility="collapsed",
            key="year_select")

    # Filter data
    df = df_all[
        (df_all["house"] == selected_house) &
        (df_all["year"] == selected_year)
    ].sort_values("day").reset_index(drop=True)

    if df.empty:
        st.warning(f"No data for {selected_house} in {selected_year}")
        st.stop()

    available_days = df["day"].dt.date.tolist()

    # Day navigation
    if "day_idx" not in st.session_state:
        st.session_state.day_idx = min(7, len(available_days) - 1)
    if st.session_state.day_idx >= len(available_days):
        st.session_state.day_idx = len(available_days) - 1

    def go_prev():
        if st.session_state.day_idx > 0:
            st.session_state.day_idx -= 1
            st.session_state._day_select = available_days[st.session_state.day_idx]

    def go_next():
        if st.session_state.day_idx < len(available_days) - 1:
            st.session_state.day_idx += 1
            st.session_state._day_select = available_days[st.session_state.day_idx]

    def on_select():
        st.session_state.day_idx = available_days.index(st.session_state._day_select)

    with filter_cols[2]:
        st.button("\u25C0 Prev", on_click=go_prev, use_container_width=True,
                  disabled=(st.session_state.day_idx == 0))
    with filter_cols[3]:
        if "_day_select" not in st.session_state:
            st.session_state._day_select = available_days[st.session_state.day_idx]
        if st.session_state._day_select not in available_days:
            st.session_state._day_select = available_days[st.session_state.day_idx]
        st.selectbox(
            "Day", available_days,
            key="_day_select",
            on_change=on_select,
            label_visibility="collapsed",
        )
    with filter_cols[4]:
        st.button("Next \u25B6", on_click=go_next, use_container_width=True,
                  disabled=(st.session_state.day_idx == len(available_days) - 1))

    selected_day = available_days[st.session_state.day_idx]

    # Monthly summary for selected month
    selected_month = selected_day.month
    month_mask = (df["day"].dt.month == selected_month)
    monthly_value = df.loc[month_mask, "battery_value"].sum()
    monthly_kwmo = monthly_value / INVERTER

    # Annual summary
    annual_value = df["battery_value"].sum()
    annual_baseline = df["cost_without_battery"].sum()

    # --- Check if interval data is available ---
    row = df[df["day"].dt.date == selected_day].iloc[0]
    has_interval_data = "actual_interval_00" in df.columns

    if has_interval_data:
        # --- Load interval data for selected day ---
        hours = np.arange(96) * 0.25
        soc_hours = np.arange(97) * 0.25

        actual = np.array([row.get(f"actual_interval_{i:02d}", 0) for i in range(96)])
        rtm = np.array([row.get(f"rtm_price_interval_{i:02d}", 0) for i in range(96)])
        charge = np.array([row.get(f"charge_interval_{i:02d}", 0) for i in range(96)])
        discharge = np.array([row.get(f"discharge_interval_{i:02d}", 0) for i in range(96)])
        soc = np.array([row.get(f"soc_interval_{i:02d}", 0) for i in range(97)])

        # --- Metrics row ---
        month_name = calendar.month_abbr[selected_month]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Daily Value", f"${row['battery_value']:.2f}")
        col2.metric(f"{month_name} Total", f"${monthly_value:.2f}")
        col3.metric(f"{month_name} $/kW-mo", f"${monthly_kwmo:.2f}")
        col4.metric(f"{selected_year} Total", f"${annual_value:.2f}")
        col5.metric("Daily Load", f"{row['load_actual_kwh']:.1f} kWh")
        col6.metric("Avg RTM Price", f"${np.mean(rtm):.1f}/MWh")

        # --- 2x2 grid of charts ---
        left, right = st.columns(2)

        with left:
            # Chart 1: Grid Import/Export with price overlay
            fig, ax1 = plt.subplots(figsize=(7, 3.5))

            # Compute net grid flow: positive = import, negative = export
            load_kw = actual / 0.25  # kWh per interval → average kW
            grid_flow_kw = load_kw + charge - discharge
            grid_import = np.maximum(grid_flow_kw, 0)
            grid_export = np.minimum(grid_flow_kw, 0)  # negative values

            ax1.bar(hours, grid_import, width=0.23, color="#4CAF50", alpha=0.8,
                    label="Grid Import")
            ax1.bar(hours, grid_export, width=0.23, color="#E53935", alpha=0.8,
                    label="Grid Export")
            # Baseline load (without battery) as reference line
            ax1.plot(hours, load_kw, color="blue", lw=1, ls="--", alpha=0.5,
                     label="Load (no battery)")
            ax1.axhline(y=0, color="black", lw=0.5)
            ax1.set_xlabel("Hour")
            ax1.set_ylabel("Power (kW)")
            ax1.set_xlim(0, 24)

            # Overlay RTM price on secondary axis (clipped to avoid spike compression)
            ax2 = ax1.twinx()
            price_cap = np.percentile(rtm, 95) * 1.5 if np.max(rtm) > np.percentile(rtm, 95) * 2 else None
            rtm_display = np.clip(rtm, 0, price_cap) if price_cap else rtm
            ax2.plot(hours, rtm_display, "darkorange", lw=1.2, alpha=0.6, label="RTM Price")
            ax2.set_ylabel("RTM Price ($/MWh)", color="darkorange")
            ax2.tick_params(axis="y", labelcolor="darkorange")
            if price_cap and np.max(rtm) > price_cap:
                ax2.annotate(f"Peak ${np.max(rtm):.0f}",
                             xy=(hours[np.argmax(rtm)], price_cap),
                             fontsize=7, color="darkorange", ha="center",
                             va="bottom", fontweight="bold")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)

            net_import_kwh = float(np.sum(grid_import) * 0.25)
            net_export_kwh = float(np.sum(-grid_export) * 0.25)
            ax1.set_title(f"Grid Flow — Import {net_import_kwh:.1f} / "
                           f"Export {net_export_kwh:.1f} kWh")
            plt.tight_layout()
            st.pyplot(fig)

        with right:
            # Chart 2: State of Charge
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(soc_hours, soc, "g-", lw=1.5)
            ax.fill_between(soc_hours, soc, alpha=0.2, color="green")
            ax.axhline(y=CAPACITY, color="red", ls="--", lw=0.8,
                       label=f"Max ({CAPACITY} kWh)")
            ax.axhline(y=CAPACITY * MIN_SOC_PCT / 100, color="orange", ls="--", lw=0.8,
                       label=f"Min SOC ({MIN_SOC_PCT}% = {CAPACITY * MIN_SOC_PCT / 100:.1f} kWh)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("SOC (kWh)")
            ax.set_xlim(0, 24)
            ax.set_ylim(0, CAPACITY * 1.1)
            ax.legend(loc="upper left")
            ax.set_title(f"State of Charge — {soc[0]:.1f} \u2192 {soc[-1]:.1f} kWh")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        left2, right2 = st.columns(2)

        with left2:
            # Chart 3: Actual Load Profile
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.fill_between(hours, actual, alpha=0.3, color="blue")
            ax.plot(hours, actual, "b-", lw=1.5, label="Actual Load")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Load (kWh/interval)")
            ax.set_xlim(0, 24)
            ax.legend(loc="upper left")
            ax.set_title(f"Actual Load Profile — {row['load_actual_kwh']:.1f} kWh total")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with right2:
            # Chart 4: RTM Prices
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(hours, rtm, "darkorange", lw=1.5, label="RTM Price")
            ax.fill_between(hours, rtm, alpha=0.15, color="orange")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Price ($/MWh)")
            ax.set_xlim(0, 24)
            ax.legend(loc="upper left")
            ax.set_title(f"RTM Settlement Prices — Mean ${np.mean(rtm):.1f}/MWh")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        # --- Summary metrics only (no interval charts) ---
        month_name = calendar.month_abbr[selected_month]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Daily Value", f"${row['battery_value']:.2f}")
        col2.metric(f"{month_name} Total", f"${monthly_value:.2f}")
        col3.metric(f"{month_name} $/kW-mo", f"${monthly_kwmo:.2f}")
        col4.metric(f"{selected_year} Total", f"${annual_value:.2f}")
        col5.metric("Daily Load", f"{row['load_actual_kwh']:.1f} kWh")

        st.info("Detailed interval charts (grid flow, SOC, load profile, RTM prices) "
                "are available in the local version of this dashboard.")

# ============================================================
# TAB: Summary (yearly chart only, no year selector)
# ============================================================
with tab_summary:
    # Weather zone -> load zone mapping
    WZ_TO_LZ = {
        "COAST": "LZ_HOUSTON", "EAST": "LZ_NORTH", "FWEST": "LZ_WEST",
        "NCENT": "LZ_NORTH", "NORTH": "LZ_NORTH", "SCENT": "LZ_SOUTH",
        "SOUTH": "LZ_SOUTH", "WEST": "LZ_WEST",
    }

    sum_metric = st.radio(
        "Metric", ["Total Value ($)", "$/kW-mo"],
        index=1,
        horizontal=True, label_visibility="collapsed",
        key="sum_metric",
    )
    sum_kwmo = sum_metric == "$/kW-mo"

    # --- Battery Value by Year and Load Zone ---
    if len(df_all["year"].unique()) > 1:
        unit_label = "$/kW-mo" if sum_kwmo else "Total Value ($)"
        st.subheader(f"Average {unit_label} by Year and Load Zone")

        df_lz = df_all.copy()
        df_lz["load_zone"] = df_lz["house"].str.rsplit("_", n=1).str[1].map(WZ_TO_LZ)

        annual_by_house = df_lz.groupby(["year", "house", "load_zone"]).agg(
            battery_value=("battery_value", "sum"),
        ).reset_index()

        if sum_kwmo:
            annual_by_house["plot_val"] = annual_by_house["battery_value"] / 12 / INVERTER
        else:
            annual_by_house["plot_val"] = annual_by_house["battery_value"]

        lz_year = annual_by_house.groupby(["year", "load_zone"])["plot_val"].mean().reset_index()
        lz_pivot = lz_year.pivot(index="year", columns="load_zone", values="plot_val")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        load_zones = sorted(lz_pivot.columns)
        lz_colors = {
            "LZ_HOUSTON": "#2196F3", "LZ_NORTH": "#4CAF50",
            "LZ_SOUTH": "#FF9800", "LZ_WEST": "#E53935",
        }
        years = lz_pivot.index.values
        n_zones = len(load_zones)
        total_width = 0.7
        bar_width = total_width / n_zones

        fmt = "${:.2f}" if sum_kwmo else "${:,.0f}"
        for i, lz in enumerate(load_zones):
            offset = (i - (n_zones - 1) / 2) * bar_width
            vals = lz_pivot[lz].values
            ax.bar(years + offset, vals, bar_width, label=lz,
                   color=lz_colors.get(lz, f"C{i}"), alpha=0.85)
            for y, v in zip(years, vals):
                ax.annotate(fmt.format(v), (y + offset, v),
                            ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xlabel("Year")
        ax.set_ylabel(unit_label)
        ax.set_title(f"Average Battery Value ({unit_label}) by Year and Load Zone")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(years)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Need multiple years of data to show the yearly summary chart.")

# ============================================================
# TAB: Details (monthly chart, sensitivity chart, house grid)
# ============================================================
with tab_details:
    # --- Metric toggle and year filter ---
    det_cols = st.columns([2, 4])
    with det_cols[0]:
        det_metric = st.radio(
            "Metric", ["Total Value ($)", "$/kW-mo"],
            index=1,
            horizontal=True, label_visibility="collapsed",
            key="det_metric",
        )
    show_kwmo = det_metric == "$/kW-mo"

    with det_cols[1]:
        year_options = ["All Years"] + [str(y) for y in available_years]
        default_year_idx = year_options.index("2025") if "2025" in year_options else 0
        details_year = st.selectbox("Filter Year", year_options,
                                    index=default_year_idx,
                                    label_visibility="collapsed",
                                    key="details_year_select")

    _WZ_TO_LZ = {
        "COAST": "LZ_HOUSTON", "EAST": "LZ_NORTH", "FWEST": "LZ_WEST",
        "NCENT": "LZ_NORTH", "NORTH": "LZ_NORTH", "SCENT": "LZ_SOUTH",
        "SOUTH": "LZ_SOUTH", "WEST": "LZ_WEST",
    }

    if details_year == "All Years":
        df_det = df_all.copy()
    else:
        df_det = df_all[df_all["year"] == int(details_year)].copy()

    n_years = df_all["year"].nunique() if details_year == "All Years" else 1
    is_all_years = details_year == "All Years"

    if df_det.empty:
        st.warning("No data for selected filter.")
    else:
        # ---- Chart 1: Monthly Value by Load Zone ----
        df_det["month_num"] = df_det["day"].dt.month
        df_det["load_zone"] = df_det["house"].str.rsplit("_", n=1).str[1].map(_WZ_TO_LZ)

        monthly_by_lz = df_det.groupby(["load_zone", "month_num", "house"]).agg(
            battery_value=("battery_value", "sum"),
        ).reset_index()
        if is_all_years:
            monthly_by_lz["battery_value"] = monthly_by_lz["battery_value"] / n_years
        monthly_by_lz = monthly_by_lz.groupby(["load_zone", "month_num"])["battery_value"].mean().reset_index()

        all_months = sorted(df_det["month_num"].unique())
        monthly_lz_pivot = monthly_by_lz.pivot(index="load_zone", columns="month_num", values="battery_value")
        monthly_lz_pivot = monthly_lz_pivot.reindex(columns=all_months)

        if show_kwmo:
            monthly_lz_pivot = monthly_lz_pivot / INVERTER

        chart_year_label = f"({details_year})" if not is_all_years else f"({n_years}-Year Average)"
        st.subheader(f"Monthly Value by Load Zone {chart_year_label}")

        fig, ax = plt.subplots(figsize=(12, 4.5))
        x = np.arange(len(all_months))
        load_zones = sorted(monthly_lz_pivot.index)
        lz_colors = {
            "LZ_HOUSTON": "#2196F3", "LZ_NORTH": "#4CAF50",
            "LZ_SOUTH": "#FF9800", "LZ_WEST": "#E53935",
        }
        n_zones = len(load_zones)
        total_width = 0.8
        bar_width = total_width / n_zones

        for i, lz in enumerate(load_zones):
            offset = (i - (n_zones - 1) / 2) * bar_width
            vals = monthly_lz_pivot.loc[lz].values
            ax.bar(x + offset, vals, bar_width, label=lz,
                   color=lz_colors.get(lz, f"C{i}"), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([calendar.month_abbr[m] for m in all_months])
        ax.set_xlabel("Month")
        ylabel = "Avg $/kW-mo per House" if show_kwmo else "Avg Monthly Value ($) per House"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Average Battery Value by Month and Load Zone {chart_year_label}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig)

        # ---- Chart 2: Battery Size Sensitivity ----
        sens_is_all = is_all_years
        sens_n_years = n_years

        sens_records = []
        for cap, path in capacity_files:
            df_cap = load_data(path)
            if not sens_is_all:
                df_cap = df_cap[df_cap["year"] == int(details_year)]
            if df_cap.empty:
                continue
            df_cap["load_zone"] = df_cap["house"].str.rsplit("_", n=1).str[1].map(_WZ_TO_LZ)
            house_vals = df_cap.groupby(["house", "load_zone"]).agg(
                battery_value=("battery_value", "sum"),
            ).reset_index()
            if sens_is_all:
                house_vals["battery_value"] = house_vals["battery_value"] / sens_n_years
            lz_avg = house_vals.groupby("load_zone")["battery_value"].mean().reset_index()
            lz_avg["capacity"] = cap
            sens_records.append(lz_avg)

        if sens_records:
            sens_df = pd.concat(sens_records, ignore_index=True)

            if show_kwmo:
                sens_df["plot_val"] = sens_df["battery_value"] / 12 / INVERTER
                sens_unit = "$/kW-mo"
            else:
                sens_df["plot_val"] = sens_df["battery_value"]
                sens_unit = "Total Value ($)"

            sens_pivot = sens_df.pivot(index="capacity", columns="load_zone", values="plot_val")
            sens_pivot = sens_pivot.sort_index()

            year_lbl = details_year if not is_all_years else f"{n_years}-Year Average"
            st.subheader(f"Average {sens_unit} by Battery Size and Load Zone — {year_lbl}")

            fig, ax = plt.subplots(figsize=(12, 5))
            load_zones_s = sorted(sens_pivot.columns)
            capacities = sens_pivot.index.values
            xs = np.arange(len(capacities))
            n_zones_s = len(load_zones_s)
            total_width_s = 0.7
            bar_width_s = total_width_s / n_zones_s

            fmt = "${:.2f}" if show_kwmo else "${:,.0f}"
            for i, lz in enumerate(load_zones_s):
                offset = (i - (n_zones_s - 1) / 2) * bar_width_s
                vals = sens_pivot[lz].values
                ax.bar(xs + offset, vals, bar_width_s, label=lz,
                       color=lz_colors.get(lz, f"C{i}"), alpha=0.85)
                for xi, v in zip(xs, vals):
                    ax.annotate(fmt.format(v), (xi + offset, v),
                                ha="center", va="bottom", fontsize=7, fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels([f"{c} kWh" for c in capacities])
            ax.set_xlabel("Battery Size (kWh)")
            ax.set_ylabel(sens_unit)
            ax.set_title(f"Average Battery Value ({sens_unit}) by Battery Size and Load Zone — {year_lbl}")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig)

        # ---- Chart 3: Pivot table (house x month grid) ----
        pivot_value = df_det.pivot_table(
            index="house", columns="month_num",
            values="battery_value", aggfunc="sum",
        )
        pivot_value = pivot_value.reindex(columns=all_months)

        if is_all_years:
            pivot_value = pivot_value / n_years

        pivot_value["Annual"] = pivot_value.sum(axis=1)

        if show_kwmo:
            pivot_display = pivot_value.copy()
            for m in all_months:
                pivot_display[m] = pivot_display[m] / INVERTER
            pivot_display["Annual"] = pivot_display["Annual"] / INVERTER / len(all_months)
        else:
            pivot_display = pivot_value.copy()

        pivot_display["Load Zone"] = [
            _WZ_TO_LZ.get(h.rsplit("_", 1)[-1], "?") for h in pivot_display.index
        ]
        pivot_display = pivot_display.reset_index().rename(columns={"house": "House"})
        pivot_display = pivot_display.sort_values(["Load Zone", "House"]).reset_index(drop=True)

        col_rename = {m: calendar.month_abbr[m] for m in all_months}
        pivot_display = pivot_display.rename(columns=col_rename)

        month_cols = [calendar.month_abbr[m] for m in all_months]
        all_display_cols = ["Load Zone", "House"] + month_cols + ["Annual"]

        styled = (
            pivot_display[all_display_cols]
            .style
            .format("${:.2f}", subset=month_cols + ["Annual"])
            .background_gradient(cmap="Greens", axis=None, subset=month_cols)
            .background_gradient(cmap="Greens", axis=None, subset=["Annual"])
            .hide(axis="index")
        )

        year_label = details_year if not is_all_years else f"Avg Annual ({n_years}-Year Average)"
        st.subheader(f"Battery Value by House and Month — {year_label}")
        if show_kwmo:
            st.caption("Values as $/kW-mo (monthly total / inverter kW). Annual = average monthly rate."
                       + (" Averaged across all years." if is_all_years else ""))
        else:
            st.caption("Values as total dollars. Annual = sum of all months."
                       + (f" Averaged across {n_years} years." if is_all_years else ""))

        st.dataframe(styled, use_container_width=True, height=620)

# ============================================================
# TAB: Methodology
# ============================================================
with tab_methodology:
    st.header("Methodology")

    st.subheader("Overview")
    st.markdown("""
This simulator calculates the **theoretical maximum battery arbitrage value**
using **perfect information** — actual realized loads and actual Real-Time Market
(RTM) settlement prices. Unlike a forecast-based simulator, the oracle optimizer
knows exactly what load and prices will be for the entire day, finding the
mathematically optimal charge/discharge schedule.

**Simulation structure:**

1. **Load actual data** — the real 15-minute load profile and RTM prices for each
   day (no forecasting).
2. **Optimize** a charge/discharge schedule using a linear program (CVXPY) that
   maximizes net revenue given perfect knowledge of load and prices.
3. **Settle** each interval at RTM prices and compute the battery's value versus
   a no-battery baseline.

The simulation covers **16 houses** (2 ERCOT residential profile types across 8
weather zones) at **15-minute resolution** (96 intervals per day) for
**2020 through 2025** (6 full calendar years).
""")

    st.divider()

    st.subheader("Battery Optimization")
    st.markdown(f"""
**Formulation.** The daily charge/discharge schedule is determined by a **linear
program** (LP) solved with CVXPY using the CLARABEL solver. The objective is to
maximize net revenue from grid transactions over the 96-interval day.

**Perfect information.** Unlike v1 (which optimized against DAM forecasts and
settled at RTM), v2 optimizes directly against **actual RTM prices** and
**actual load**. This produces the theoretical maximum battery value — the upper
bound on what any strategy could achieve.

**Zone-specific pricing.** Each house uses RTM prices from its ERCOT load zone
(LZ_HOUSTON, LZ_NORTH, LZ_SOUTH, or LZ_WEST), not a statewide average.

**Asymmetric pricing.** Grid imports are priced at RTM + T&D costs
(${TD_COSTS:.0f}/MWh), while exports are at RTM wholesale. This makes
self-consumption more valuable than grid export.

**Decision variables:**
- `C[t]` — charge power (kW) at each interval
- `D[t]` — discharge power (kW) at each interval
- `S[t]` — state of charge (kWh) at each interval boundary
- `grid_import[t]`, `grid_export[t]` — power flows at the site meter (kW)

**Constraints:**

| Constraint | Description |
|-----------|-------------|
| SOC dynamics | S[t+1] = S[t] + C[t] * dt * eta_c - D[t] * dt / eta_d |
| SOC bounds | {CAPACITY * MIN_SOC_PCT / 100:.0f} kWh <= S[t] <= {CAPACITY} kWh |
| Inverter limits | C[t] <= {INVERTER} kW, D[t] <= {INVERTER} kW |
| No simultaneous charge/discharge | C[t] + D[t] <= {INVERTER} kW |
| Export cap | grid_export[t] <= max({INVERTER} - load[t], 0) |
| Energy balance | grid_import[t] + D[t] = load[t] + C[t] + grid_export[t] |

**Efficiency.** Round-trip efficiency ({EFFICIENCY:.0%}) is split symmetrically:
charge efficiency = discharge efficiency = sqrt({EFFICIENCY}) = {np.sqrt(EFFICIENCY):.4f}.
""")

    st.divider()

    st.subheader("Settlement")
    st.markdown(f"""
**Cost with battery** = sum of (grid_flow_kWh x RTM_price) across all intervals.
Positive grid flow is an import (expense); negative is an export (revenue).

**Cost without battery** = sum of (actual_load_kWh x RTM_price) — the
counterfactual where all load is served directly by the grid.

**Battery value** = cost_without - cost_with. This represents the wholesale
energy savings created by the battery's optimal charge/discharge behavior.

**Note on T&D adder.** The ${TD_COSTS:.0f}/MWh retail adder is used by the
optimizer to make economically correct charge/discharge decisions (imports are
more expensive than exports), but both cost_with and cost_without are settled at
wholesale RTM prices. This means the reported battery value captures wholesale
energy savings only and does not include the additional T&D cost savings from
avoided grid imports. The true consumer savings would be somewhat higher.

**SOC management.** State of charge carries forward day-to-day within each year.
At year boundaries, SOC resets to 50% of capacity so each year is independently
comparable.
""")

    st.divider()

    st.subheader("Key Assumptions")
    st.markdown(f"""
- **Perfect information.** The optimizer sees actual RTM prices and actual load
  for the entire day. This is the theoretical upper bound — no real controller
  can achieve this.
- **Single-day optimization horizon.** The battery schedule is optimized one day
  at a time (96 intervals). While state of charge carries forward between days,
  the optimizer has no visibility into future days' prices or load. A multi-day
  optimizer could potentially extract additional value by deferring
  charge/discharge across days.
- **No grid outage modeling.** The simulation assumes the grid is always
  available for both importing and exporting power. Actual grid outages would
  eliminate both import and export capability during outage periods, reducing
  realized battery value. This is particularly relevant for extreme weather
  events (e.g., Winter Storm Uri in February 2021) where the model may
  overstate achievable value.
- **No market participation fees.** Only wholesale energy and T&D costs are
  modeled; no ancillary services, demand charges, or retail rate structures.
- **Flat T&D adder.** T&D costs are a constant ${TD_COSTS:.0f}/MWh on imports.
  Actual TDSP charges vary by utility and time of use.
- **No battery degradation.** The battery maintains full capacity throughout
  the 6-year simulation period. In practice, capacity fade would reduce
  achievable value in later years.
- **No ramp constraints.** The battery can switch between full charge and full
  discharge instantaneously.
- **Price-taker.** The battery's actions do not affect market prices.
""")

    st.divider()

    st.subheader("Weather Zone to Load Zone Mapping")
    st.markdown("""
| Weather Zone | Load Zone | Houses |
|---|---|---|
| COAST | LZ_HOUSTON | RESHIWR_COAST, RESLOWR_COAST |
| EAST | LZ_NORTH | RESHIWR_EAST, RESLOWR_EAST |
| NCENT | LZ_NORTH | RESHIWR_NCENT, RESLOWR_NCENT |
| NORTH | LZ_NORTH | RESHIWR_NORTH, RESLOWR_NORTH |
| FWEST | LZ_WEST | RESHIWR_FWEST, RESLOWR_FWEST |
| WEST | LZ_WEST | RESHIWR_WEST, RESLOWR_WEST |
| SCENT | LZ_SOUTH | RESHIWR_SCENT, RESLOWR_SCENT |
| SOUTH | LZ_SOUTH | RESHIWR_SOUTH, RESLOWR_SOUTH |
""")

    st.divider()

    st.subheader("Data Sources")
    st.markdown("""
| Dataset | Resolution | Period |
|---------|------------|--------|
| ERCOT Backcasted Load Profiles — Residential | 15-min | 2020–2025 |
| ERCOT RTM Settlement Point Prices (per load zone) | 15-min | 2020–2025 |

All data sourced from ERCOT's publicly available archives.
""")
