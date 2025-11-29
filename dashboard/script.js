// Dashboard Logic

// Configuration
const CONFIG = {
    colors: {
        primary: '#3b82f6',
        secondary: '#8b5cf6',
        success: '#10b981',
        danger: '#ef4444',
        warning: '#f59e0b',
        grid: '#334155',
        text: '#94a3b8'
    },
    chartDefaults: {
        chart: {
            background: 'transparent',
            toolbar: { show: false },
            animations: { enabled: true }
        },
        theme: { mode: 'dark' },
        grid: {
            borderColor: '#334155',
            strokeDashArray: 4
        },
        dataLabels: { enabled: false }
    }
};

// State
let state = {
    backtestData: [],
    portfolioData: [],
    detailedData: [],
    metrics: {}
};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log("Starting dashboard initialization...");
        await loadData();
        console.log("Data loaded successfully");
        renderDashboard();
        console.log("Dashboard rendered");
    } catch (error) {
        console.error('Failed to load dashboard:', error);
        // Show error state in UI
        const container = document.querySelector('.container') || document.body;
        const errorDiv = document.createElement('div');
        errorDiv.style.color = 'red';
        errorDiv.style.padding = '20px';
        errorDiv.style.background = 'rgba(255,0,0,0.1)';
        errorDiv.innerHTML = `<h3>Dashboard Error</h3><p>${error.message}</p><pre>${error.stack}</pre>`;
        container.prepend(errorDiv);
    }
});

async function loadData() {
    // Load Backtest Results
    try {
        console.log("Fetching backtest_results.csv...");
        const backtestResponse = await fetch('backtest_results.csv');  // Try same dir first
        if (!backtestResponse.ok) {
            // Try parent directory
            const parentResponse = await fetch('../backtest_results.csv');
            if (!parentResponse.ok) throw new Error(`Failed to fetch backtest_results.csv: ${parentResponse.status}`);
            const backtestText = await parentResponse.text();
            console.log("Backtest data received from parent dir, length:", backtestText.length);
            Papa.parse(backtestText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    state.backtestData = results.data;
                    console.log("Parsed backtest data rows:", state.backtestData.length);
                    calculateMetrics(results.data);
                },
                error: (err) => { throw new Error(`CSV Parse Error: ${err.message}`); }
            });
        } else {
            const backtestText = await backtestResponse.text();
            console.log("Backtest data received from same dir, length:", backtestText.length);
            Papa.parse(backtestText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    state.backtestData = results.data;
                    console.log("Parsed backtest data rows:", state.backtestData.length);
                    calculateMetrics(results.data);
                },
                error: (err) => { throw new Error(`CSV Parse Error: ${err.message}`); }
            });
        }
    } catch (e) {
        console.error("Error loading backtest data:", e);
        throw e;
    }

    // Load Final Portfolio
    try {
        console.log("Fetching final_portfolio.csv...");
        let portfolioResponse = await fetch('final_portfolio.csv');
        if (!portfolioResponse.ok) {
            portfolioResponse = await fetch('../final_portfolio.csv');
        }
        if (portfolioResponse.ok) {
            const portfolioText = await portfolioResponse.text();
            const lines = portfolioText.split('\n').filter(l => l.trim());
            state.portfolioData = lines.map(line => {
                const [name, value] = line.split(',');
                return { name: name || 'Unknown', value: parseFloat(value) };
            }).filter(item => item.name && !isNaN(item.value) && item.value > 0.001);
            console.log("Parsed portfolio data items:", state.portfolioData.length);
        }
    } catch (e) {
        console.warn("Error loading final_portfolio.csv:", e);
    }

    // Load Detailed Portfolio (if available)
    try {
        console.log("Fetching final_portfolio_detailed.csv...");
        let detailedResponse = await fetch('final_portfolio_detailed.csv');
        if (!detailedResponse.ok) {
            detailedResponse = await fetch('../final_portfolio_detailed.csv');
        }
        if (detailedResponse.ok) {
            const detailedText = await detailedResponse.text();
            const dLines = detailedText.split('\n').filter(l => l.trim());
            state.detailedData = dLines.map(line => {
                const [name, value] = line.split(',');
                return { name: name || 'Unknown', value: parseFloat(value) };
            }).filter(item => item.name && !isNaN(item.value) && item.value > 0.001)
                .sort((a, b) => b.value - a.value);
            console.log("Parsed detailed data items:", state.detailedData.length);
        } else {
            console.log("final_portfolio_detailed.csv not found");
        }
    } catch (e) {
        console.log("Detailed portfolio not found or error:", e);
    }
}

function calculateMetrics(data) {
    if (!data.length) return;

    // Assuming data has 'Portfolio_Value' or 'Cumulative_Return'
    // We'll calculate simple metrics from the time series if pre-calculated ones aren't available

    const startValue = data[0]['Value'] || 100000;
    const endValue = data[data.length - 1]['Value'];
    const totalReturn = ((endValue - startValue) / startValue) * 100;

    // Calculate Drawdown
    let peak = -Infinity;
    let maxDrawdown = 0;

    data.forEach(row => {
        const val = row['Value'];
        if (val > peak) peak = val;
        const dd = (peak - val) / peak;
        if (dd > maxDrawdown) maxDrawdown = dd;
    });

    // Calculate Volatility (Annualized)
    // Simple approximation using daily returns
    const returns = [];
    for (let i = 1; i < data.length; i++) {
        const r = (data[i]['Value'] - data[i - 1]['Value']) / data[i - 1]['Value'];
        returns.push(r);
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    const annualizedVol = stdDev * Math.sqrt(252); // Assuming daily data

    // Sharpe (assuming 2% risk free)
    const rf = 0.02;
    const annualizedReturn = (Math.pow((endValue / startValue), 252 / data.length) - 1);
    const sharpe = (annualizedReturn - rf) / annualizedVol;

    state.metrics = {
        totalReturn: totalReturn,
        maxDrawdown: maxDrawdown * 100,
        volatility: annualizedVol * 100,
        sharpe: sharpe
    };
}

function renderDashboard() {
    renderMetrics();
    renderPerformanceChart();
    renderAllocationChart();
    renderDrawdownChart();
    renderDrawdownChart();
    renderMonthlyReturnsChart();
    renderHoldingsTable();
}

function renderMetrics() {
    document.getElementById('metric-return').textContent = `${state.metrics.totalReturn.toFixed(2)}%`;
    document.getElementById('metric-return').style.color = state.metrics.totalReturn >= 0 ? CONFIG.colors.success : CONFIG.colors.danger;

    document.getElementById('metric-sharpe').textContent = state.metrics.sharpe.toFixed(2);

    document.getElementById('metric-drawdown').textContent = `-${state.metrics.maxDrawdown.toFixed(2)}%`;

    document.getElementById('metric-volatility').textContent = `${state.metrics.volatility.toFixed(2)}%`;
}

function renderPerformanceChart() {
    const data = state.backtestData;
    const options = {
        ...CONFIG.chartDefaults,
        series: [{
            name: 'Portfolio Value',
            data: data.map(d => ({
                x: new Date(d.Date).getTime(),
                y: d.Value
            }))
        }],
        chart: {
            type: 'area',
            height: 350,
            fontFamily: 'Inter, sans-serif',
            background: 'transparent',
            toolbar: { show: false }
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.1,
                stops: [0, 90, 100]
            }
        },
        xaxis: {
            type: 'datetime',
            tooltip: { enabled: false },
            axisBorder: { show: false },
            axisTicks: { show: false }
        },
        yaxis: {
            labels: {
                formatter: (value) => {
                    return '$' + value.toLocaleString(undefined, { maximumFractionDigits: 0 });
                }
            }
        },
        colors: [CONFIG.colors.primary]
    };

    const chart = new ApexCharts(document.querySelector("#performanceChart"), options);
    chart.render();
}

function renderAllocationChart() {
    const data = state.portfolioData;

    const options = {
        ...CONFIG.chartDefaults,
        series: data.map(d => d.value),
        labels: data.map(d => d.name.replace(/_/g, ' ')),
        chart: {
            type: 'donut',
            height: 350,
            background: 'transparent'
        },
        plotOptions: {
            pie: {
                donut: {
                    size: '70%',
                    labels: {
                        show: true,
                        name: { color: CONFIG.colors.text },
                        value: { color: '#fff', fontSize: '24px', fontWeight: 600 },
                        total: {
                            show: true,
                            label: 'Total',
                            color: CONFIG.colors.text,
                            formatter: () => '100%'
                        }
                    }
                }
            }
        },
        stroke: { show: false },
        legend: {
            position: 'bottom',
            labels: { colors: CONFIG.colors.text }
        },
        colors: [
            '#3b82f6', '#8b5cf6', '#10b981', '#f59e0b',
            '#ef4444', '#ec4899', '#6366f1', '#14b8a6'
        ]
    };

    const chart = new ApexCharts(document.querySelector("#allocationChart"), options);
    chart.render();
}

function renderDrawdownChart() {
    // Calculate drawdown series
    let peak = -Infinity;
    const drawdownSeries = state.backtestData.map(d => {
        const val = d.Value;
        if (val > peak) peak = val;
        const dd = ((val - peak) / peak) * 100;
        return {
            x: new Date(d.Date).getTime(),
            y: dd
        };
    });

    const options = {
        ...CONFIG.chartDefaults,
        series: [{
            name: 'Drawdown',
            data: drawdownSeries
        }],
        chart: {
            type: 'area',
            height: 300,
            background: 'transparent',
            toolbar: { show: false }
        },
        stroke: {
            curve: 'straight',
            width: 1
        },
        fill: {
            type: 'solid',
            opacity: 0.2
        },
        xaxis: {
            type: 'datetime'
        },
        yaxis: {
            labels: {
                formatter: (val) => val.toFixed(1) + '%'
            }
        },
        colors: [CONFIG.colors.danger]
    };

    const chart = new ApexCharts(document.querySelector("#drawdownChart"), options);
    chart.render();
}

function renderMonthlyReturnsChart() {
    // Calculate monthly returns from monthly data points
    const monthlyData = [];

    for (let i = 1; i < state.backtestData.length; i++) {
        const current = state.backtestData[i];
        const prev = state.backtestData[i - 1];

        const ret = ((current.Value - prev.Value) / prev.Value) * 100;
        const date = new Date(current.Date);
        const label = `${date.getFullYear()}-${date.getMonth() + 1}`; // format for chart

        monthlyData.push({
            x: label,
            y: ret
        });
    }

    // Take last 12
    const recentMonths = monthlyData.slice(-12);

    const options = {
        ...CONFIG.chartDefaults,
        series: [{
            name: 'Monthly Return',
            data: recentMonths.map(d => d.y)
        }],
        chart: {
            type: 'bar',
            height: 300,
            background: 'transparent',
            toolbar: { show: false }
        },
        plotOptions: {
            bar: {
                borderRadius: 4,
                colors: {
                    ranges: [{
                        from: -100,
                        to: 0,
                        color: CONFIG.colors.danger
                    }, {
                        from: 0,
                        to: 100,
                        color: CONFIG.colors.success
                    }]
                }
            }
        },
        xaxis: {
            categories: recentMonths.map(d => d.x)
        },
        yaxis: {
            labels: {
                formatter: (val) => val.toFixed(1) + '%'
            }
        }
    };

    const chart = new ApexCharts(document.querySelector("#monthlyReturnsChart"), options);
    chart.render();
}

function renderHoldingsTable() {
    const tbody = document.getElementById('holdingsTableBody');
    if (!tbody) return;

    const data = state.detailedData.length > 0 ? state.detailedData : state.portfolioData;

    tbody.innerHTML = data.map(item => `
        <tr>
            <td>${item.name.replace(/_/g, ' ')}</td>
            <td>${(item.value * 100).toFixed(2)}%</td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${Math.min(item.value * 100, 100)}%"></div>
                </div>
            </td>
        </tr>
    `).join('');
}
