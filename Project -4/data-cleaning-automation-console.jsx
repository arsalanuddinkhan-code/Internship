import React, { useState, useMemo } from "react";
import {
  BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip as RTooltip, ResponsiveContainer, Legend,
} from "recharts";
import {
  Workflow, Database, ShieldCheck, Sparkles, FileCheck2, CheckCircle2,
  ArrowRight, TerminalSquare, TrendingUp, AlertCircle, Copy, Globe2, Tags,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/*  EMBEDDED DATA (produced by data_cleaning_automation.py)            */
/* ------------------------------------------------------------------ */
const DATA = {"summary":{"rows_before":1622,"rows_after":1600,"duplicates_removed":22,"missing_before":253,"missing_after":0,"quality_score_before":98.6,"quality_score_after":100.0,"total_revenue":456327.76,"avg_order_value":285.2,"return_rate":0.089},"log":[{"time":"03:59:51","stage":"Ingest","message":"Loading raw file: orders_raw.csv","count":null},{"time":"03:59:51","stage":"Ingest","message":"Loaded 1622 rows, 11 columns","count":null},{"time":"03:59:51","stage":"Validate","message":"Missing values by column (before): Email=64, Country=50, ProductCategory=32, Quantity=24, Discount=51, PaymentMethod=32","count":null},{"time":"03:59:51","stage":"Validate","message":"Exact duplicate rows found","count":22},{"time":"03:59:51","stage":"Clean","message":"Removed exact duplicate rows","count":22},{"time":"03:59:51","stage":"Standardize","message":"'ProductCategory': normalized 1251 inconsistent values to canonical labels","count":1251},{"time":"03:59:51","stage":"Standardize","message":"'Country': normalized 1211 inconsistent values to canonical labels","count":1211},{"time":"03:59:51","stage":"Standardize","message":"'OrderDate': parsed 1600 mixed-format date strings into a single format","count":null},{"time":"03:59:51","stage":"Standardize","message":"'UnitPrice': converted 1600 string-formatted currency values to numeric","count":1600},{"time":"03:59:51","stage":"Standardize","message":"'IsReturned': collapsed 8 representations (Yes/Y/true/1, etc.) into True/False","count":null},{"time":"03:59:51","stage":"Clean","message":"Flagged and nulled 11 impossible Discount value(s) (outside 0-100%)","count":11},{"time":"03:59:51","stage":"Impute","message":"'ProductCategory': filled 32 missing value(s) with mode ('Home & Garden')","count":32},{"time":"03:59:51","stage":"Impute","message":"'Country': filled 48 missing value(s) with mode ('United Kingdom')","count":48},{"time":"03:59:51","stage":"Impute","message":"'PaymentMethod': filled 32 missing value(s) with 'Unknown'","count":32},{"time":"03:59:51","stage":"Impute","message":"'Quantity': filled 24 missing value(s) with median (3.0)","count":24},{"time":"03:59:51","stage":"Impute","message":"'Discount': filled 59 missing value(s) (incl. flagged outliers) with median (0.15)","count":59},{"time":"03:59:51","stage":"Impute","message":"'Email': filled 64 missing value(s) with placeholder","count":64},{"time":"03:59:51","stage":"Validate","message":"Missing values remaining (after)","count":0},{"time":"03:59:51","stage":"Validate","message":"Duplicate rows remaining (after)","count":0},{"time":"03:59:51","stage":"Report","message":"Data quality score before cleaning: 98.6/100","count":null},{"time":"03:59:51","stage":"Report","message":"Data quality score after cleaning: 100.0/100","count":null}],"missingByColumn":[{"column":"Country","before":50,"after":0},{"column":"Discount","before":51,"after":0},{"column":"Email","before":64,"after":0},{"column":"PaymentMethod","before":32,"after":0},{"column":"ProductCategory","before":32,"after":0},{"column":"Quantity","before":24,"after":0}],"byCategory":[{"category":"Apparel","revenue":89581.46,"orders":310},{"category":"Beauty","revenue":89737.49,"orders":315},{"category":"Electronics","revenue":82577.49,"orders":296},{"category":"Home & Garden","revenue":103390.67,"orders":362},{"category":"Sports","revenue":91040.65,"orders":317}],"byCountry":[{"country":"Australia","revenue":88653.23,"orders":328},{"country":"Canada","revenue":82311.79,"orders":299},{"country":"Germany","revenue":92135.31,"orders":314},{"country":"United Kingdom","revenue":115028.21,"orders":382},{"country":"United States","revenue":78199.22,"orders":277}],"samples":[{"orderId":"ORD-21390","beforeCategory":" Apparel","beforeCountry":"CANADA","beforeDate":"07-May-2026","beforePrice":"70.76"},{"orderId":"ORD-20002","beforeCategory":"Apparel","beforeCountry":"DE","beforeDate":"26-Jun-2025","beforePrice":"$122.24"},{"orderId":"ORD-20503","beforeCategory":"Home & Garden","beforeCountry":"AUS","beforeDate":"April 07, 2025","beforePrice":"$197.06"},{"orderId":"ORD-21340","beforeCategory":"electronics","beforeCountry":"United kingdom","beforeDate":"2026-01-31","beforePrice":"24.27"},{"orderId":"ORD-21025","beforeCategory":"Sports ","beforeCountry":"USA","beforeDate":"09/17/2025","beforePrice":"$149.32"},{"orderId":"ORD-20472","beforeCategory":"Sport","beforeCountry":"America","beforeDate":"2025-12-09","beforePrice":"$51.30"}],"stages":[{"key":"ingest","label":"Ingest","detail":"1622 rows loaded"},{"key":"validate","label":"Validate","detail":"253 missing, 22 dupes found"},{"key":"clean","label":"Clean","detail":"22 dupes removed"},{"key":"standardize","label":"Standardize","detail":"categories, dates, currency, booleans"},{"key":"impute","label":"Impute","detail":"253 values filled"},{"key":"report","label":"Report","detail":"quality 98.6\u2192100.0"}]};
const { summary, log, missingByColumn, byCategory, byCountry, samples, stages } = DATA;

// small mirrors of the pipeline's canonical maps, for the live "before -> after" demo
const CATEGORY_MAP = {
  "electronics": "Electronics", "electronic": "Electronics",
  "apparel": "Apparel", "appreal": "Apparel",
  "home & garden": "Home & Garden", "home and garden": "Home & Garden", "home&garden": "Home & Garden",
  "sports": "Sports", "sport": "Sports",
  "beauty": "Beauty", "beuaty": "Beauty",
};
const COUNTRY_MAP = {
  "united states": "United States", "usa": "United States", "u.s.a.": "United States",
  "us": "United States", "america": "United States",
  "canada": "Canada", "ca": "Canada",
  "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom",
  "england": "United Kingdom",
  "australia": "Australia", "aus": "Australia",
  "germany": "Germany", "de": "Germany",
};
const standardize = (val, map) => map[String(val).trim().toLowerCase()] || val;
const cleanPrice = (val) => {
  const n = parseFloat(String(val).replace(/[^0-9.\-]/g, ""));
  return isNaN(n) ? val : `$${n.toFixed(2)}`;
};

/* ------------------------------------------------------------------ */
/*  TOKENS                                                              */
/* ------------------------------------------------------------------ */
const C = {
  bg: "#EEF2F7",
  panel: "#FFFFFF",
  panelAlt: "#F4F7FB",
  border: "#D8E0EA",
  borderSoft: "#E4EAF1",
  ink: "#17233A",
  inkDim: "#586A85",
  inkFaint: "#93A2B8",
  blue: "#2E6F9E",
  blueDeep: "#1F4E73",
  amber: "#D98E3B",
  green: "#2E9E63",
  red: "#D1524B",
  console: "#111722",
  consoleBorder: "#232E42",
};
const STAGE_COLORS = { ingest: C.blue, validate: C.amber, clean: C.red, standardize: C.blueDeep, impute: C.green, report: C.green };
const CHART_COLORS = [C.blue, C.green, C.amber, C.blueDeep, C.red, "#7C6FC9"];

const FONT_IMPORT =
  "@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');";

const fmtCurrency = (n) => "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmtNumber = (n) => Number(n).toLocaleString();
const fmtPct = (n) => `${(n * 100).toFixed(1)}%`;

/* ------------------------------------------------------------------ */
/*  MAIN                                                                */
/* ------------------------------------------------------------------ */
export default function AutomationConsole() {
  const [activeStage, setActiveStage] = useState(null);
  const [copiedRow, setCopiedRow] = useState(null);

  const filteredLog = useMemo(() => {
    if (!activeStage) return log;
    return log.filter((l) => l.stage.toLowerCase() === activeStage);
  }, [activeStage]);

  const qualityDelta = summary.quality_score_after - summary.quality_score_before;
  const missingResolved = summary.missing_before - summary.missing_after;

  const copyRow = (id) => {
    setCopiedRow(id);
    setTimeout(() => setCopiedRow((v) => (v === id ? null : v)), 1200);
  };

  return (
    <div className="dc-root">
      <style>{`
        ${FONT_IMPORT}
        .dc-root { background: ${C.bg}; color: ${C.ink}; font-family: 'Inter', sans-serif; min-height: 100vh; width: 100%;
          background-image: linear-gradient(${C.borderSoft} 1px, transparent 1px), linear-gradient(90deg, ${C.borderSoft} 1px, transparent 1px);
          background-size: 28px 28px; }
        .dc-root * { box-sizing: border-box; }
        .dc-display { font-family: 'Manrope', sans-serif; }
        .dc-mono { font-family: 'JetBrains Mono', monospace; }

        .dc-header { padding: 26px 28px 18px; }
        .dc-eyebrow { font-size: 11px; letter-spacing: 0.13em; text-transform: uppercase; color: ${C.blue}; display: flex; align-items: center; gap: 7px; margin-bottom: 8px; font-weight: 700; }
        .dc-title { font-size: 26px; font-weight: 800; }
        .dc-subtitle { font-size: 12.5px; color: ${C.inkDim}; margin-top: 6px; max-width: 620px; line-height: 1.5; }

        .dc-body { padding: 6px 28px 46px; max-width: 1360px; margin: 0 auto; }

        /* Pipeline runway */
        .dc-runway { background: ${C.panel}; border: 1px solid ${C.border}; border-radius: 10px; padding: 18px 20px; margin-bottom: 20px; overflow-x: auto; }
        .dc-runway-track { display: flex; align-items: center; gap: 4px; min-width: 640px; }
        .dc-stage { flex: 1; display: flex; flex-direction: column; align-items: flex-start; gap: 6px; padding: 10px 12px; border-radius: 8px; cursor: pointer; border: 1px solid transparent; transition: all .12s ease; }
        .dc-stage:hover { background: ${C.panelAlt}; }
        .dc-stage.active { background: ${C.panelAlt}; border-color: ${C.blue}; }
        .dc-stage-top { display: flex; align-items: center; gap: 7px; }
        .dc-stage-num { width: 20px; height: 20px; border-radius: 50%; background: ${C.green}; color: #fff; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
        .dc-stage-label { font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 12.5px; }
        .dc-stage-detail { font-size: 10.5px; color: ${C.inkFaint}; font-family: 'JetBrains Mono', monospace; }
        .dc-arrow { color: ${C.borderSoft}; flex-shrink: 0; }

        /* KPI strip */
        .dc-kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 20px; }
        .dc-kpi { background: ${C.panel}; border: 1px solid ${C.border}; border-radius: 8px; padding: 14px 16px; }
        .dc-kpi-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: ${C.inkFaint}; margin-bottom: 6px; display: flex; align-items: center; gap: 5px; }
        .dc-kpi-value { font-family: 'Manrope', sans-serif; font-size: 21px; font-weight: 800; }
        .dc-kpi-sub { font-size: 11px; color: ${C.green}; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

        /* Panels */
        .dc-panel { background: ${C.panel}; border: 1px solid ${C.border}; border-radius: 10px; padding: 18px; margin-bottom: 20px; }
        .dc-panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; flex-wrap: wrap; gap: 8px; }
        .dc-panel-title { font-family: 'Manrope', sans-serif; font-size: 14.5px; font-weight: 700; display: flex; align-items: center; gap: 8px; }
        .dc-panel-sub { font-size: 11px; color: ${C.inkFaint}; }

        .dc-grid-2 { display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; }

        /* Log console */
        .dc-console { background: ${C.console}; border: 1px solid ${C.consoleBorder}; border-radius: 8px; padding: 14px 16px; max-height: 320px; overflow-y: auto; }
        .dc-log-line { display: flex; gap: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; padding: 4px 0; border-bottom: 1px solid #1B2434; }
        .dc-log-line:last-child { border-bottom: none; }
        .dc-log-time { color: #5A6A85; flex-shrink: 0; }
        .dc-log-stage { flex-shrink: 0; width: 78px; font-weight: 600; }
        .dc-log-msg { color: #C7D0DE; }
        .dc-log-count { color: #7EE0A8; margin-left: auto; flex-shrink: 0; padding-left: 10px; }

        /* Before/after samples */
        .dc-sample-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .dc-sample-table th { text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: ${C.inkFaint}; padding: 7px 10px; border-bottom: 1px solid ${C.border}; }
        .dc-sample-table td { padding: 8px 10px; border-bottom: 1px solid ${C.borderSoft}; vertical-align: middle; }
        .dc-transform { display: flex; align-items: center; gap: 7px; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; }
        .dc-before { color: ${C.red}; text-decoration: line-through; text-decoration-color: ${C.red}66; }
        .dc-after { color: ${C.green}; font-weight: 600; }
        .dc-copy-btn { background: ${C.panelAlt}; border: 1px solid ${C.border}; border-radius: 5px; padding: 3px 7px; cursor: pointer; color: ${C.inkDim}; font-size: 10px; display: flex; align-items: center; gap: 4px; }

        /* Missing value bars */
        .dc-missing-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .dc-missing-label { width: 110px; font-size: 11.5px; color: ${C.inkDim}; flex-shrink: 0; }
        .dc-missing-bar-wrap { flex: 1; background: ${C.borderSoft}; border-radius: 4px; height: 16px; position: relative; overflow: hidden; }
        .dc-missing-bar { height: 100%; background: ${C.red}; opacity: 0.35; position: absolute; left: 0; top: 0; }
        .dc-missing-bar.after { background: ${C.green}; opacity: 0.55; }
        .dc-missing-val { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; width: 30px; text-align: right; flex-shrink: 0; }

        @media (max-width: 980px) {
          .dc-kpi-grid { grid-template-columns: repeat(2, 1fr); }
          .dc-grid-2 { grid-template-columns: 1fr; }
          .dc-runway-track { min-width: 560px; }
        }
        @media (max-width: 560px) {
          .dc-header { padding: 20px 16px 14px; }
          .dc-body { padding: 4px 16px 32px; }
          .dc-kpi-grid { grid-template-columns: 1fr 1fr; }
        }
      `}</style>

      <div className="dc-header">
        <div className="dc-eyebrow"><Workflow size={13} /> DATA OPS · CLEANING &amp; REPORTING AUTOMATION</div>
        <div className="dc-title dc-display">Pipeline run report</div>
        <div className="dc-subtitle">
          Output of data_cleaning_automation.py: an idempotent pipeline that ingests a raw export, validates it,
          cleans and standardizes every field, imputes what's missing, and generates this report automatically.
        </div>
      </div>

      <div className="dc-body">
        {/* Pipeline runway */}
        <div className="dc-runway">
          <div className="dc-runway-track">
            {stages.map((s, i) => (
              <React.Fragment key={s.key}>
                <div className={`dc-stage ${activeStage === s.key ? "active" : ""}`} onClick={() => setActiveStage(activeStage === s.key ? null : s.key)}>
                  <div className="dc-stage-top">
                    <div className="dc-stage-num"><CheckCircle2 size={13} /></div>
                    <div className="dc-stage-label">{s.label}</div>
                  </div>
                  <div className="dc-stage-detail">{s.detail}</div>
                </div>
                {i < stages.length - 1 && <ArrowRight size={16} className="dc-arrow" />}
              </React.Fragment>
            ))}
          </div>
        </div>

        {/* KPI strip */}
        <div className="dc-kpi-grid">
          <Kpi icon={<ShieldCheck size={12} />} label="Quality score" value={`${summary.quality_score_after}/100`} sub={`+${qualityDelta.toFixed(1)} from ${summary.quality_score_before}`} />
          <Kpi icon={<Database size={12} />} label="Rows processed" value={fmtNumber(summary.rows_after)} sub={`${summary.duplicates_removed} duplicates removed`} />
          <Kpi icon={<AlertCircle size={12} />} label="Missing values resolved" value={fmtNumber(missingResolved)} sub={`${summary.missing_after} remaining`} />
          <Kpi icon={<TrendingUp size={12} />} label="Total revenue" value={fmtCurrency(summary.total_revenue)} sub={`avg order ${fmtCurrency(summary.avg_order_value)}`} />
          <Kpi icon={<FileCheck2 size={12} />} label="Return rate" value={fmtPct(summary.return_rate)} sub="post-cleaning" />
        </div>

        {/* Cleaning log console */}
        <div className="dc-panel">
          <div className="dc-panel-header">
            <div className="dc-panel-title"><TerminalSquare size={16} /> Cleaning Log</div>
            <div className="dc-panel-sub">{activeStage ? `Filtered: ${activeStage}` : "Click a pipeline stage above to filter"}</div>
          </div>
          <div className="dc-console">
            {filteredLog.map((l, i) => (
              <div className="dc-log-line" key={i}>
                <span className="dc-log-time">{l.time}</span>
                <span className="dc-log-stage" style={{ color: STAGE_COLORS[l.stage.toLowerCase()] || "#8FA0BE" }}>{l.stage}</span>
                <span className="dc-log-msg">{l.message}</span>
                {l.count !== null && <span className="dc-log-count">+{l.count}</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Missing values before/after + Before/after samples */}
        <div className="dc-grid-2">
          <div className="dc-panel">
            <div className="dc-panel-header">
              <div className="dc-panel-title"><AlertCircle size={16} /> Missing Values by Column</div>
              <div className="dc-panel-sub">before → after</div>
            </div>
            {missingByColumn.map((m) => {
              const max = Math.max(...missingByColumn.map((x) => x.before));
              return (
                <div className="dc-missing-row" key={m.column}>
                  <div className="dc-missing-label">{m.column}</div>
                  <div className="dc-missing-bar-wrap">
                    <div className="dc-missing-bar" style={{ width: `${(m.before / max) * 100}%` }} />
                  </div>
                  <div className="dc-missing-val">{m.before}</div>
                  <ArrowRight size={12} color={C.inkFaint} />
                  <div className="dc-missing-bar-wrap" style={{ maxWidth: 60 }}>
                    <div className="dc-missing-bar after" style={{ width: m.after > 0 ? `${(m.after / max) * 100}%` : "2px" }} />
                  </div>
                  <div className="dc-missing-val">{m.after}</div>
                </div>
              );
            })}
          </div>

          <div className="dc-panel">
            <div className="dc-panel-header">
              <div className="dc-panel-title"><Tags size={16} /> Standardization Preview</div>
              <div className="dc-panel-sub">live transform, same rules as the pipeline</div>
            </div>
            <table className="dc-sample-table">
              <thead><tr><th>Order</th><th>Category</th><th>Country</th><th>Price</th><th></th></tr></thead>
              <tbody>
                {samples.map((s) => {
                  const afterCat = standardize(s.beforeCategory, CATEGORY_MAP);
                  const afterCountry = standardize(s.beforeCountry, COUNTRY_MAP);
                  const afterPrice = cleanPrice(s.beforePrice);
                  return (
                    <tr key={s.orderId}>
                      <td className="dc-mono" style={{ fontSize: 11 }}>{s.orderId}</td>
                      <td><div className="dc-transform"><span className="dc-before">{s.beforeCategory.trim() || "—"}</span><ArrowRight size={11} color={C.inkFaint} /><span className="dc-after">{afterCat}</span></div></td>
                      <td><div className="dc-transform"><span className="dc-before">{s.beforeCountry}</span><ArrowRight size={11} color={C.inkFaint} /><span className="dc-after">{afterCountry}</span></div></td>
                      <td><div className="dc-transform"><span className="dc-before">{s.beforePrice}</span><ArrowRight size={11} color={C.inkFaint} /><span className="dc-after">{afterPrice}</span></div></td>
                      <td>
                        <button className="dc-copy-btn" onClick={() => copyRow(s.orderId)}>
                          <Copy size={10} /> {copiedRow === s.orderId ? "noted" : "flag"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Revenue breakdowns */}
        <div className="dc-grid-2">
          <div className="dc-panel">
            <div className="dc-panel-header">
              <div className="dc-panel-title"><Sparkles size={16} /> Revenue by Category</div>
              <div className="dc-panel-sub">cleaned &amp; standardized data</div>
            </div>
            <ResponsiveContainer width="100%" height={230}>
              <PieChart>
                <Pie data={byCategory} dataKey="revenue" nameKey="category" innerRadius={46} outerRadius={78} paddingAngle={2} stroke={C.panel} strokeWidth={2}>
                  {byCategory.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Pie>
                <RTooltip formatter={(v) => fmtCurrency(v)} contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="dc-panel">
            <div className="dc-panel-header">
              <div className="dc-panel-title"><Globe2 size={16} /> Orders by Country</div>
              <div className="dc-panel-sub">country names standardized</div>
            </div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={byCountry} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid stroke={C.borderSoft} vertical={false} />
                <XAxis dataKey="country" tick={{ fill: C.inkFaint, fontSize: 10 }} axisLine={{ stroke: C.border }} tickLine={false} />
                <YAxis tick={{ fill: C.inkFaint, fontSize: 10.5 }} axisLine={false} tickLine={false} width={30} />
                <RTooltip contentStyle={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 12 }} />
                <Bar dataKey="orders" radius={[4, 4, 0, 0]}>
                  {byCountry.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function Kpi({ icon, label, value, sub }) {
  return (
    <div className="dc-kpi">
      <div className="dc-kpi-label">{icon} {label}</div>
      <div className="dc-kpi-value">{value}</div>
      {sub && <div className="dc-kpi-sub">{sub}</div>}
    </div>
  );
}
