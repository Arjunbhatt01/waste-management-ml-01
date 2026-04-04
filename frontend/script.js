/**
 * ═══════════════════════════════════════════════════════
 * AI Smart Waste Management System - Dehradun
 * Frontend JavaScript
 * ═══════════════════════════════════════════════════════
 *
 * Handles:
 *  - API calls to Flask backend
 *  - Chart.js visualizations (Pie, Bar, Line, Doughnut)
 *  - Leaflet map rendering for Dehradun waste zones
 *  - AI waste classification form
 *  - Citizen report form submission
 *  - Dashboard tab navigation
 *  - Toast notification system
 */

'use strict';

// ───────────────────────────────────────────
// Configuration
// ───────────────────────────────────────────
const API_BASE = window.location.origin;  // Flask server origin
const IS_DASHBOARD = window.location.pathname.includes('dashboard');

// ───────────────────────────────────────────
// Chart.js Global Defaults
// Sets up color theme and font for all charts
// ───────────────────────────────────────────
if (typeof Chart !== 'undefined') {
  Chart.defaults.color = '#8b9bb4';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0f2034';
  Chart.defaults.plugins.tooltip.borderColor = 'rgba(0,212,170,0.3)';
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 12;
  Chart.defaults.plugins.tooltip.titleColor = '#f0f6fc';
  Chart.defaults.plugins.tooltip.bodyColor = '#8b9bb4';
}

// ───────────────────────────────────────────
// Color Palettes
// ───────────────────────────────────────────
const WASTE_COLORS = {
  Plastic:  '#3b82f6',
  Organic:  '#22c55e',
  Metal:    '#f59e0b',
  Paper:    '#8b5cf6',
  Glass:    '#06b6d4',
};

const AREA_COLORS = [
  '#00d4aa', '#7c3aed', '#f59e0b', '#3b82f6', '#ef4444'
];

const WASTE_ICONS = {
  Plastic: '🔵', Organic: '🟢', Metal: '🟡', Paper: '🟣', Glass: '🔷'
};

// ───────────────────────────────────────────
// State
// ───────────────────────────────────────────
let charts = {};      // Holds Chart.js instances (for re-rendering)
let dashMap = null;   // Leaflet map instance for dashboard
let homeMap = null;   // Leaflet map instance for home page

// ───────────────────────────────────────────
// Toast Notification System
// ───────────────────────────────────────────

/**
 * Shows a toast notification at the bottom right.
 * @param {string} message - The text to display
 * @param {'success'|'error'|'info'} type - Visual style
 * @param {number} duration - Auto-dismiss after ms (default 4000)
 */
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const icons = { success: '✅', error: '❌', info: 'ℹ️' };
  const toast = document.createElement('div');
  toast.className = `toast-item ${type}`;
  toast.innerHTML = `
    <span style="font-size:1.1rem;">${icons[type]}</span>
    <span style="font-size:0.88rem;color:var(--text-primary);">${message}</span>
    <button onclick="this.parentElement.remove()" style="margin-left:auto;background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:1rem;">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => { if (toast.parentNode) toast.remove(); }, duration);
}

// ───────────────────────────────────────────
// Tab Navigation (Dashboard)
// ───────────────────────────────────────────

/**
 * Switches between dashboard sections/tabs.
 * Lazy-loads map when map tab first opened.
 * @param {string} tabName - Tab identifier
 */
function showTab(tabName) {
  // Hide all tabs
  document.querySelectorAll('.dash-tab').forEach(t => t.classList.remove('active'));
  // Remove active class from all sidebar links
  document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));

  // Show target tab
  const tab = document.getElementById(`tab-${tabName}`);
  if (tab) tab.classList.add('active');

  // Activate sidebar link
  const link = document.getElementById(`link-${tabName}`);
  if (link) link.classList.add('active');

  // Update top bar title
  const titles = {
    overview: 'Overview', analytics: 'Analytics',
    predictions: '7-Day Predictions', schedule: 'Collection Schedule',
    reports: 'Citizen Reports', map: 'Waste Map', classifier: 'AI Classifier'
  };
  const titleEl = document.getElementById('page-title');
  if (titleEl) titleEl.textContent = titles[tabName] || tabName;

  // Lazy-init dashboard map on first open
  if (tabName === 'map' && !dashMap) {
    setTimeout(() => initDashboardMap(), 100);
  }

  // Close mobile sidebar
  closeSidebar();
}

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  if (sidebar.classList.contains('open')) {
    closeSidebar();
  } else {
    sidebar.classList.add('open');
    overlay.classList.add('open');
  }
}

function closeSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  if (sidebar) sidebar.classList.remove('open');
  if (overlay) overlay.classList.remove('open');
}

// ───────────────────────────────────────────
// Dashboard Data Loader
// ───────────────────────────────────────────

/**
 * Main function to load all dashboard data.
 * Called on page load. Fetches dashboard stats and predictions
 * from Flask API and renders all charts.
 */
async function loadDashboard() {
  try {
    // Parallel API requests for speed
    const [dashRes, predRes] = await Promise.all([
      fetch(`${API_BASE}/get_dashboard_data`),
      fetch(`${API_BASE}/get_predictions`)
    ]);

    const dashData = await dashRes.json();
    const predData = await predRes.json();

    if (dashData.status !== 'success') throw new Error('Dashboard data failed');
    if (predData.status !== 'success') throw new Error('Predictions failed');

    // Render all sections
    renderKPICards(dashData);
    renderPieChart(dashData.waste_by_type);
    renderBarChart(dashData.waste_by_area);
    renderSchedule(dashData.collection_schedule, 'quick-schedule', 3);  // Quick view: 3 items
    renderSchedule(dashData.collection_schedule, 'full-schedule-container');  // Full view
    renderReportsDonut(dashData.reports_summary);
    renderVolumeBarChart(dashData.waste_by_area);
    renderRecyclableChart(dashData.stats);
    renderHeatmap(dashData.waste_by_type, dashData.waste_by_area);
    renderForecastChart(predData);
    renderPredictionTable(predData);
    renderPredKPIs(predData);

    // Update timestamp
    const timeEl = document.getElementById('last-updated-time');
    if (timeEl) timeEl.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;

  } catch (err) {
    console.error('Dashboard load error:', err);
    showToast('Could not load dashboard data. Is Flask server running?', 'error');
  }
}

// ───────────────────────────────────────────
// KPI Cards Renderer
// ───────────────────────────────────────────

/**
 * Renders the 5 KPI stat cards at the top of the dashboard.
 * Includes total waste, volume, recyclable %, reports, and accuracy.
 */
function renderKPICards(data) {
  const stats = data.stats;
  const kpiContainer = document.getElementById('kpi-row');
  if (!kpiContainer) return;

  const recyclePct = stats.total_records > 0
    ? Math.round((stats.recyclable_count / stats.total_records) * 100)
    : 0;

  const kpis = [
    {
      icon: '⚖️', color: 'teal',
      value: stats.total_waste_kg ? `${(stats.total_waste_kg / 1000).toFixed(1)}T` : '--',
      label: 'Total Waste', change: '↑ 4.2% vs last week', changeType: 'up'
    },
    {
      icon: '🧪', color: 'blue',
      value: stats.total_volume ? `${stats.total_volume.toFixed(0)}L` : '--',
      label: 'Total Volume', change: '↑ 2.8% vs last week', changeType: 'up'
    },
    {
      icon: '♻️', color: 'green',
      value: `${recyclePct}%`,
      label: 'Recyclable Rate', change: '↑ 1.5% improvement', changeType: 'up'
    },
    {
      icon: '🚩', color: 'amber',
      value: data.total_reports,
      label: 'Citizen Reports', change: 'Real-time tracking', changeType: ''
    },
    {
      icon: '🤖', color: 'purple',
      value: '95%',
      label: 'ML Accuracy', change: 'Random Forest model', changeType: ''
    },
  ];

  kpiContainer.innerHTML = `
    <div class="row g-3">
      ${kpis.map(k => `
        <div class="col-xl col-lg-4 col-md-4 col-6">
          <div class="stat-card">
            <div class="stat-icon ${k.color}">${k.icon}</div>
            <div>
              <div class="stat-value">${k.value}</div>
              <div class="stat-label">${k.label}</div>
              ${k.change ? `<div class="stat-change ${k.changeType}">${k.change}</div>` : ''}
            </div>
          </div>
        </div>
      `).join('')}
    </div>
  `;

  // Also update home page hero quick stats
  const totalWasteStat = document.getElementById('stat-total-waste');
  if (totalWasteStat && stats.total_waste_kg) {
    totalWasteStat.textContent = (stats.total_waste_kg / 1000).toFixed(1);
  }
}

// ───────────────────────────────────────────
// Chart Renderers
// ───────────────────────────────────────────

/**
 * Pie chart: Waste type distribution by weight.
 * Why Pie: Best for showing categorical proportions at a glance.
 */
function renderPieChart(data) {
  const ctx = document.getElementById('pieChart');
  if (!ctx) return;
  if (charts.pie) charts.pie.destroy();

  charts.pie = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => d.waste_type),
      datasets: [{
        data: data.map(d => d.total_kg),
        backgroundColor: data.map(d => WASTE_COLORS[d.waste_type] || '#666'),
        borderColor: '#0f2034',
        borderWidth: 3,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} kg`
          }
        }
      }
    }
  });
}

/**
 * Bar chart: Area-wise waste generation.
 * Why Bar: Effective for comparing discrete categories (areas).
 */
function renderBarChart(data) {
  const ctx = document.getElementById('barChart');
  if (!ctx) return;
  if (charts.bar) charts.bar.destroy();

  charts.bar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.area),
      datasets: [{
        label: 'Weight (kg)',
        data: data.map(d => d.total_kg),
        backgroundColor: AREA_COLORS.map(c => c + 'cc'),
        borderColor: AREA_COLORS,
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b9bb4' },
          beginAtZero: true
        },
        x: {
          grid: { display: false },
          ticks: { color: '#8b9bb4', maxRotation: 0 }
        }
      }
    }
  });
}

/**
 * Volume bar chart for analytics tab.
 * Shows area-wise liters instead of kg for a different metric view.
 */
function renderVolumeBarChart(data) {
  const ctx = document.getElementById('volumeBarChart');
  if (!ctx) return;
  if (charts.volumeBar) charts.volumeBar.destroy();

  charts.volumeBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.area),
      datasets: [{
        label: 'Volume (Liters)',
        data: data.map(d => d.total_volume),
        backgroundColor: 'rgba(0,212,170,0.2)',
        borderColor: '#00d4aa',
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      }, {
        label: 'Collections Count',
        data: data.map(d => d.collections * 10),
        backgroundColor: 'rgba(124,58,237,0.2)',
        borderColor: '#7c3aed',
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
        type: 'line',
        fill: false,
        tension: 0.4,
        pointRadius: 5,
        yAxisID: 'y2',
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top' } },
      scales: {
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b9bb4' }, beginAtZero: true },
        y2: { position: 'right', grid: { display: false }, ticks: { display: false } },
        x: { grid: { display: false }, ticks: { color: '#8b9bb4' } }
      }
    }
  });
}

/**
 * Recyclable vs Non-recyclable doughnut chart.
 */
function renderRecyclableChart(stats) {
  const ctx = document.getElementById('recyclableChart');
  if (!ctx || !stats.total_records) return;
  if (charts.recyclable) charts.recyclable.destroy();

  const recyclable = stats.recyclable_count || 0;
  const nonRecyclable = stats.total_records - recyclable;

  charts.recyclable = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Recyclable', 'Non-Recyclable'],
      datasets: [{
        data: [recyclable, nonRecyclable],
        backgroundColor: ['rgba(0,212,170,0.7)', 'rgba(239,68,68,0.7)'],
        borderColor: ['#00d4aa', '#ef4444'],
        borderWidth: 2,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.raw} records` } }
      }
    }
  });
}

/**
 * Reports status donut chart.
 * Shows pending / resolved / in_progress distribution.
 */
function renderReportsDonut(data) {
  const ctx = document.getElementById('reportsDonut');
  if (!ctx) return;
  if (charts.reportsDonut) charts.reportsDonut.destroy();

  const statusColors = {
    pending: '#f59e0b', resolved: '#22c55e', in_progress: '#3b82f6'
  };

  charts.reportsDonut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: data.map(d => d.status.charAt(0).toUpperCase() + d.status.slice(1)),
      datasets: [{
        data: data.map(d => d.count),
        backgroundColor: data.map(d => statusColors[d.status] || '#666'),
        borderColor: '#0f2034',
        borderWidth: 3,
        hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '60%',
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

/**
 * 7-day forecast multi-line chart.
 * Why Line Chart: Best for showing trends over time per category.
 * Each area gets its own colored line.
 */
function renderForecastChart(predData) {
  const ctx = document.getElementById('forecastChart');
  if (!ctx) return;
  if (charts.forecast) charts.forecast.destroy();

  const dateLabels = predData.dates.map(d => {
    const date = new Date(d);
    return date.toLocaleDateString('en-IN', { weekday: 'short', month: 'short', day: 'numeric' });
  });

  const datasets = predData.areas.map((area, i) => ({
    label: area,
    data: predData.predictions[area],
    borderColor: AREA_COLORS[i],
    backgroundColor: AREA_COLORS[i] + '20',
    fill: true,
    tension: 0.4,
    pointRadius: 5,
    pointHoverRadius: 8,
    borderWidth: 2.5,
  }));

  charts.forecast = new Chart(ctx, {
    type: 'line',
    data: { labels: dateLabels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.dataset.label}: ${c.raw} L`
          }
        }
      },
      scales: {
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b9bb4', callback: v => `${v}L` },
          beginAtZero: false
        },
        x: { grid: { display: false }, ticks: { color: '#8b9bb4' } }
      }
    }
  });
}

// ───────────────────────────────────────────
// Schedule Renderer
// ───────────────────────────────────────────

/**
 * Renders collection schedule cards.
 * Priority is color-coded (High=red, Medium=amber, Low=green).
 * @param {Array} schedule - Array of schedule objects
 * @param {string} containerId - Target element ID
 * @param {number} limit - Max items to show (default: all)
 */
function renderSchedule(schedule, containerId, limit = 99) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const items = schedule.slice(0, limit);

  container.innerHTML = items.map((item, i) => `
    <div class="schedule-row">
      <div class="priority-indicator priority-${item.priority}" title="Priority: ${item.priority}"></div>
      <div class="schedule-time ms-3">${item.time}</div>
      <div class="ms-3" style="flex:1;">
        <div style="font-weight:600;font-size:0.9rem;color:var(--text-primary);">${item.area}</div>
        <div style="font-size:0.75rem;color:var(--text-secondary);">Est. ${item.estimated_volume} L · Priority: ${item.priority}</div>
      </div>
      <div>
        <span class="badge-custom" style="background:rgba(0,212,170,0.1);color:var(--primary);">
          ${i + 1}st Run
        </span>
      </div>
    </div>
  `).join('');
}

// ───────────────────────────────────────────
// Prediction Table Renderer
// ───────────────────────────────────────────

function renderPredictionTable(predData) {
  const container = document.getElementById('predictions-table-container');
  if (!container) return;

  const headers = ['Area', ...predData.dates.map(d =>
    new Date(d).toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric' })
  )];

  container.innerHTML = `
    <div style="overflow-x:auto;">
      <table style="width:100%;border-collapse:separate;border-spacing:0 4px;">
        <thead>
          <tr>
            ${headers.map(h => `
              <th style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.06em;color:var(--text-secondary);padding:0.6rem 0.75rem;
                background:rgba(0,0,0,0.2);text-align:center;">
                ${h}
              </th>
            `).join('')}
          </tr>
        </thead>
        <tbody>
          ${predData.areas.map(area => `
            <tr>
              <td style="padding:0.7rem 0.75rem;font-weight:600;color:var(--text-primary);
                background:var(--bg-card);border-top:1px solid var(--border);white-space:nowrap;">
                ${area}
              </td>
              ${predData.predictions[area].map(vol => {
                const pct = Math.min(100, (vol / 80) * 100);
                const color = pct > 70 ? '#ef4444' : pct > 45 ? '#f59e0b' : '#22c55e';
                return `
                  <td style="padding:0.7rem 0.5rem;text-align:center;background:var(--bg-card);
                    border-top:1px solid var(--border);" title="${vol} L">
                    <div style="font-size:0.82rem;font-weight:700;color:${color};">${vol}L</div>
                    <div style="width:100%;height:3px;background:rgba(255,255,255,0.05);
                      border-radius:2px;margin-top:4px;overflow:hidden;">
                      <div style="width:${pct}%;height:100%;background:${color};border-radius:2px;"></div>
                    </div>
                  </td>
                `;
              }).join('')}
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function renderPredKPIs(predData) {
  const container = document.getElementById('pred-kpi-row');
  if (!container) return;

  // Calculate total predicted for day 1
  let day1Total = 0;
  predData.areas.forEach(a => { day1Total += predData.predictions[a][0] || 0; });

  // Find peak day
  let peakDay = 0, peakVol = 0;
  for (let d = 0; d < 7; d++) {
    let dayTotal = 0;
    predData.areas.forEach(a => { dayTotal += predData.predictions[a][d] || 0; });
    if (dayTotal > peakVol) { peakVol = dayTotal; peakDay = d; }
  }

  const peakDate = new Date(predData.dates[peakDay]).toLocaleDateString('en-IN', { weekday: 'long' });

  container.innerHTML = `
    <div class="row g-3">
      <div class="col-md-4">
        <div class="stat-card">
          <div class="stat-icon teal">📅</div>
          <div>
            <div class="stat-value">${day1Total.toFixed(0)}L</div>
            <div class="stat-label">Today's Prediction</div>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="stat-card">
          <div class="stat-icon amber">📈</div>
          <div>
            <div class="stat-value">${peakDate}</div>
            <div class="stat-label">Peak Day This Week</div>
            <div class="stat-change up">↑ ${peakVol.toFixed(0)}L predicted</div>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="stat-card">
          <div class="stat-icon purple">🤖</div>
          <div>
            <div class="stat-value">87%</div>
            <div class="stat-label">Forecast Confidence</div>
            <div class="stat-change">RF Regressor Model</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ───────────────────────────────────────────
// Heatmap Renderer (Analytics Tab)
// ───────────────────────────────────────────

function renderHeatmap(byType, byArea) {
  const table = document.getElementById('heatmap-table');
  if (!table) return;

  const areas = byArea.map(a => a.area);
  const types = byType.map(t => t.waste_type);

  // Build simple simulated cross-table data
  const heatData = {};
  areas.forEach(area => {
    heatData[area] = {};
    types.forEach(type => {
      // Simulate proportional distribution
      const areaData = byArea.find(a => a.area === area);
      const typeData = byType.find(t => t.waste_type === type);
      const proportion = (areaData?.total_kg || 0) * (typeData?.total_kg || 0) / 1000;
      heatData[area][type] = Math.round(proportion * 0.1 * 10) / 10;
    });
  });

  const maxVal = Math.max(...Object.values(heatData).flatMap(a => Object.values(a)));

  table.innerHTML = `
    <thead>
      <tr>
        <th style="padding:0.5rem;color:var(--text-secondary);font-size:0.75rem;">Area / Type</th>
        ${types.map(t => `
          <th style="padding:0.5rem;text-align:center;color:var(--text-secondary);font-size:0.75rem;font-weight:700;">
            ${WASTE_ICONS[t] || ''} ${t}
          </th>
        `).join('')}
      </tr>
    </thead>
    <tbody>
      ${areas.map(area => `
        <tr>
          <td style="padding:0.5rem;font-weight:600;color:var(--text-primary);font-size:0.85rem;white-space:nowrap;">${area}</td>
          ${types.map(type => {
            const val = heatData[area][type];
            const intensity = maxVal > 0 ? val / maxVal : 0;
            const bg = `rgba(0,212,170,${0.05 + intensity * 0.55})`;
            const textColor = intensity > 0.6 ? '#00d4aa' : '#8b9bb4';
            return `
              <td style="padding:0.5rem;text-align:center;background:${bg};
                border-radius:4px;font-weight:600;color:${textColor};font-size:0.82rem;">
                ${val}
              </td>
            `;
          }).join('')}
        </tr>
      `).join('')}
    </tbody>
  `;
}

// ───────────────────────────────────────────
// Reports Loader
// ───────────────────────────────────────────

/**
 * Fetches citizen reports from API and renders reports table.
 */
async function loadReports() {
  const tbody = document.getElementById('reports-tbody');
  if (!tbody) return;

  try {
    const res = await fetch(`${API_BASE}/get_reports`);
    const data = await res.json();

    if (!data.reports || data.reports.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:2rem;color:var(--text-muted);">No reports yet</td></tr>';
      return;
    }

    const statusClass = { pending: 'status-pending', resolved: 'status-resolved', in_progress: 'status-in-progress' };

    tbody.innerHTML = data.reports.map(r => `
      <tr>
        <td>#${r.id}</td>
        <td style="font-weight:500;">${r.reporter_name}</td>
        <td>${r.location}</td>
        <td>
          <span class="badge-custom badge-${r.waste_type.toLowerCase()}">${r.waste_type}</span>
        </td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
          ${r.description || '—'}
        </td>
        <td>
          <span class="badge-custom ${statusClass[r.status] || 'status-pending'}">
            ${r.status.replace('_', ' ')}
          </span>
        </td>
        <td style="white-space:nowrap;color:var(--text-secondary);">${formatDate(r.created_at)}</td>
      </tr>
    `).join('');

  } catch (err) {
    console.error('Reports load error:', err);
    if (tbody) tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--danger);">Error loading reports</td></tr>';
  }
}

function formatDate(dateStr) {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: '2-digit', hour: '2-digit', minute: '2-digit' });
}

// ───────────────────────────────────────────
// AI Waste Classification (Home Page)
// ───────────────────────────────────────────

/**
 * Calls /predict_waste API and displays classification result.
 * Uses Random Forest Classifier on backend.
 */
async function predictWaste() {
  const btn = document.getElementById('btn-predict');
  const resultBox = document.getElementById('prediction-result');
  if (!resultBox) return;

  // Loading state
  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Classifying...';
  resultBox.innerHTML = `
    <div class="skeleton" style="width:60px;height:60px;border-radius:50%;margin:0 auto 1rem;"></div>
    <div class="skeleton" style="width:120px;height:24px;border-radius:4px;margin:0 auto 0.5rem;"></div>
    <div class="skeleton" style="width:180px;height:16px;border-radius:4px;margin:0 auto;"></div>
  `;

  try {
    const weight   = document.getElementById('pred-weight')?.value || 2.5;
    const moisture = document.getElementById('pred-moisture')?.value || 15;
    const material = document.getElementById('pred-material')?.value || 'synthetic_polymer';
    const recEl    = document.querySelector('input[name="pred-recyclable"]:checked');
    const recyclable = recEl ? recEl.value : '1';

    const res = await fetch(`${API_BASE}/predict_waste`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weight_kg: +weight, moisture_pct: +moisture, recyclable: +recyclable, material_type: material })
    });

    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Prediction failed');

    displayPredictionResult(data, resultBox);
  } catch (err) {
    resultBox.innerHTML = `
      <div style="color:var(--danger);font-size:1.5rem;">❌</div>
      <div style="color:var(--danger);margin-top:0.5rem;font-size:0.9rem;">${err.message || 'Error occurred'}</div>
    `;
    showToast('Prediction failed. Is Flask server running?', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-brain"></i> Classify Waste';
  }
}

/**
 * Dashboard version of waste predictor.
 */
async function dashboardPredict() {
  const btn = document.getElementById('btn-dash-predict');
  const resultBox = document.getElementById('dash-prediction-result');
  if (!resultBox) return;

  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Classifying...';
  resultBox.innerHTML = `<div style="color:var(--text-muted);">Analyzing...</div>`;

  try {
    const weight   = document.getElementById('d-pred-weight')?.value || 2.5;
    const moisture = document.getElementById('d-pred-moisture')?.value || 15;
    const material = document.getElementById('d-pred-material')?.value || 'synthetic_polymer';
    const recEl    = document.querySelector('input[name="d-pred-rec"]:checked');
    const recyclable = recEl ? recEl.value : '1';

    const res = await fetch(`${API_BASE}/predict_waste`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weight_kg: +weight, moisture_pct: +moisture, recyclable: +recyclable, material_type: material })
    });

    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error);

    displayPredictionResult(data, resultBox);
  } catch (err) {
    resultBox.innerHTML = `<div style="color:var(--danger);">Error: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-brain"></i> Run Classification';
  }
}

/**
 * Renders the visual prediction result card.
 * Shows waste type, confidence bar, recommendation action.
 */
function displayPredictionResult(data, container) {
  const icons = { Plastic: '🔵', Organic: '🟢', Metal: '🟡', Paper: '🟣', Glass: '🔷' };
  const actionColors = { Recycle: '#3b82f6', Compost: '#22c55e', Landfill: '#ef4444' };
  const confPct = Math.round(data.confidence * 100);

  container.innerHTML = `
    <div class="result-icon">${icons[data.waste_type] || '♻️'}</div>
    <div class="result-type" style="color:${data.color || 'var(--primary)'};">${data.waste_type}</div>
    <div style="margin:0.75rem 0 0.5rem;font-size:0.8rem;color:var(--text-secondary);">Confidence</div>
    <div style="width:80%;background:rgba(255,255,255,0.08);border-radius:50px;height:8px;margin:0 auto 0.5rem;">
      <div style="width:${confPct}%;height:100%;background:${data.color || '#00d4aa'};border-radius:50px;
        transition:width 0.8s ease;box-shadow: 0 0 8px ${data.color || '#00d4aa'};"></div>
    </div>
    <div class="result-confidence">${confPct}% confident</div>
    <div class="result-action mt-3" style="background:${actionColors[data.recommendation] || '#3b82f6'}22;
      color:${actionColors[data.recommendation] || '#3b82f6'};
      border:1px solid ${actionColors[data.recommendation] || '#3b82f6'}44;border-radius:50px;">
      ${data.recommendation === 'Recycle' ? '♻️' : data.recommendation === 'Compost' ? '🌱' : '🗑️'}
      ${data.recommendation}
    </div>
    <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:1rem;max-width:200px;text-align:center;line-height:1.5;">
      ${data.segregation_tip}
    </div>
  `;

  showToast(`Classified as ${data.waste_type} — ${data.recommendation} recommended!`, 'success');
}

// ───────────────────────────────────────────
// Report Form Submission
// ───────────────────────────────────────────

/**
 * Submits citizen garbage report to /report_waste endpoint.
 * Supports text fields + optional image file upload (multipart form).
 */
async function submitReport(event) {
  event.preventDefault();
  const btn = document.getElementById('btn-submit-report');
  const originalText = btn.innerHTML;

  btn.disabled = true;
  btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Submitting...';

  try {
    const formData = new FormData();
    formData.append('reporter_name', document.getElementById('rep-name')?.value || '');
    formData.append('location', document.getElementById('rep-location')?.value || '');
    formData.append('waste_type', document.getElementById('rep-waste-type')?.value || 'Mixed');
    formData.append('description', document.getElementById('rep-description')?.value || '');

    const imageFile = document.getElementById('rep-image')?.files?.[0];
    if (imageFile) formData.append('image', imageFile);

    const res = await fetch(`${API_BASE}/report_waste`, {
      method: 'POST',
      body: formData  // No Content-Type header; browser sets multipart boundary automatically
    });

    const data = await res.json();
    if (data.status !== 'success') throw new Error(data.error || 'Submission failed');

    showToast(data.message, 'success', 5000);

    // Reset form
    document.getElementById('report-form')?.reset();
    const fileLabel = document.getElementById('file-label');
    if (fileLabel) fileLabel.textContent = 'Click to upload photo (optional)';

  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalText;
  }
}

function handleFileSelect(input) {
  const label = document.getElementById('file-label');
  if (label && input.files?.[0]) {
    label.textContent = `📎 ${input.files[0].name}`;
  }
}

// ───────────────────────────────────────────
// Leaflet Map (Home Page)
// ───────────────────────────────────────────

/**
 * Dehradun waste zone map for the home page.
 * Uses Leaflet.js with OpenStreetMap tiles.
 * Marker size = waste volume indicator.
 */
function initHomeMap() {
  const mapEl = document.getElementById('waste-map');
  if (!mapEl || homeMap) return;

  homeMap = L.map('waste-map', {
    center: [30.3165, 78.0322],  // Dehradun coordinates
    zoom: 12,
    zoomControl: true,
  });

  // Dark-style tiles
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap, © CartoDB',
    maxZoom: 18
  }).addTo(homeMap);

  addWasteMarkers(homeMap);
}

/**
 * Dashboard map — same setup as home map but with additional info.
 */
function initDashboardMap() {
  const mapEl = document.getElementById('dashboard-map');
  if (!mapEl || dashMap) return;

  dashMap = L.map('dashboard-map', {
    center: [30.3165, 78.0322],
    zoom: 12,
  });

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap, © CartoDB',
    maxZoom: 18
  }).addTo(dashMap);

  addWasteMarkers(dashMap, true);
}

/**
 * Adds colored circle markers for each Dehradun waste zone.
 * Color and radius reflect waste volume level.
 */
function addWasteMarkers(mapInstance, detailed = false) {
  const zones = [
    { name: 'Clock Tower', lat: 30.3247, lng: 78.0413, volume: 65, color: '#ef4444', priority: 'High' },
    { name: 'ISBT',        lat: 30.2921, lng: 78.0492, volume: 55, color: '#f59e0b', priority: 'High' },
    { name: 'Rajpur Road', lat: 30.3559, lng: 78.0637, volume: 45, color: '#f59e0b', priority: 'Medium' },
    { name: 'Prem Nagar',  lat: 30.2875, lng: 78.0161, volume: 38, color: '#22c55e', priority: 'Medium' },
    { name: 'Clement Town',lat: 30.2701, lng: 78.0213, volume: 32, color: '#22c55e', priority: 'Low' },
  ];

  zones.forEach(zone => {
    const radius = 300 + zone.volume * 10;  // Scale marker by volume

    // Outer glow circle
    L.circle([zone.lat, zone.lng], {
      radius: radius * 1.5,
      color: zone.color,
      fillColor: zone.color,
      fillOpacity: 0.08,
      weight: 1,
    }).addTo(mapInstance);

    // Main circle marker
    const circle = L.circle([zone.lat, zone.lng], {
      radius: radius,
      color: zone.color,
      fillColor: zone.color,
      fillOpacity: 0.35,
      weight: 2,
    }).addTo(mapInstance);

    // Popup with waste info
    const popupContent = `
      <div style="font-family:Inter,sans-serif;color:#f0f6fc;min-width:180px;">
        <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem;color:${zone.color};">
          📍 ${zone.name}
        </div>
        <div style="font-size:0.82rem;line-height:1.8;">
          <b>Volume:</b> ~${zone.volume} L/day<br/>
          <b>Priority:</b> ${zone.priority}<br/>
          ${detailed ? `<b>Status:</b> Active collection<br/>` : ''}
        </div>
        ${detailed ? `
          <div style="margin-top:0.5rem;padding:0.4rem 0.7rem;border-radius:4px;
            background:${zone.color}22;color:${zone.color};font-size:0.75rem;font-weight:600;">
            Next Collection: ${zone.priority === 'High' ? '6:00 AM' : zone.priority === 'Medium' ? '8:00 AM' : '10:00 AM'}
          </div>
        ` : ''}
      </div>
    `;
    circle.bindPopup(popupContent, {
      className: 'custom-popup',
      maxWidth: 220
    });

    // Label marker
    const icon = L.divIcon({
      className: '',
      html: `<div style="
        background:rgba(5,11,24,0.85);
        border:1px solid ${zone.color}66;
        color:${zone.color};
        padding:2px 8px;
        border-radius:4px;
        font-size:0.72rem;
        font-weight:700;
        font-family:Inter,sans-serif;
        white-space:nowrap;
      ">${zone.name}</div>`,
      iconAnchor: [50, 0]
    });
    L.marker([zone.lat, zone.lng], { icon }).addTo(mapInstance);
  });
}

// ───────────────────────────────────────────
// Page Initializations
// ───────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  if (IS_DASHBOARD) {
    // Dashboard page: load full dashboard data
    if (typeof loadDashboard === 'function') {
      loadDashboard();
      loadReports();
    }
    // Show overview tab by default
    showTab('overview');
  } else {
    // Home page: init map, load quick stats
    initHomeMap();

    // Also fetch quick stats for hero section
    fetch(`${API_BASE}/get_dashboard_data`)
      .then(r => r.json())
      .then(data => {
        if (data.stats?.total_waste_kg) {
          const el = document.getElementById('stat-total-waste');
          if (el) el.textContent = (data.stats.total_waste_kg / 1000).toFixed(1);
        }
      })
      .catch(() => {});  // Silently fail on home page
  }

  // Navbar scroll effect
  const navbar = document.getElementById('mainNavbar');
  if (navbar) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        navbar.style.background = 'rgba(5,11,24,0.97)';
      } else {
        navbar.style.background = 'rgba(5,11,24,0.9)';
      }
    }, { passive: true });
  }
});

// ───────────────────────────────────────────
// Leaflet popup custom styles (injected)
// ───────────────────────────────────────────
const mapPopupStyle = document.createElement('style');
mapPopupStyle.textContent = `
  .custom-popup .leaflet-popup-content-wrapper {
    background: #0d1b2a !important;
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }
  .custom-popup .leaflet-popup-tip {
    background: #0d1b2a !important;
  }
`;
document.head.appendChild(mapPopupStyle);
