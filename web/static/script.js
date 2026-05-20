/* ====================================================================
   BioRec System - Front-end JavaScript Suite
   ==================================================================== */

// App State Management
let currentMethod = "gnn";
let currentTab = "recommender";
let autocompleteData = { genes: [], drugs: [] };
let graphData = { nodes: [], links: [] };

// Canvas Force-Directed Engine State
const canvas = document.getElementById("network-canvas");
const ctx = canvas.getContext("2d");
let animationFrameId = null;
let scale = 1.0;
let offsetX = 0;
let offsetY = 0;
let isDraggingCanvas = false;
let startX = 0;
let startY = 0;
let draggedNode = null;
let hoveredNode = null;

// Physics Parameters
const repulsionConstant = 600;
const springConstant = 0.05;
const springLength = 100;
const friction = 0.85;
const centerGravity = 0.02;

// Chart.js Instances
let rocChart = null;
let prChart = null;

// ====================================================================
// Tab Architecture & Navigation
// ====================================================================
function switchTab(tabName) {
    currentTab = tabName;
    
    // Toggle active nav buttons
    document.getElementById("btn-recommender").classList.toggle("active", tabName === "recommender");
    document.getElementById("btn-validation").classList.toggle("active", tabName === "validation");
    
    // Toggle active panels
    document.getElementById("tab-recommender").classList.toggle("active", tabName === "recommender");
    document.getElementById("tab-validation").classList.toggle("active", tabName === "validation");
    
    if (tabName === "validation") {
        loadValidationDashboard();
    }
}

function selectMethod(method) {
    currentMethod = method;
    document.getElementById("method-gnn").classList.toggle("active", method === "gnn");
    document.getElementById("method-svd").classList.toggle("active", method === "svd");
    
    // Retrigger search if there is a query already present
    const inputVal = document.getElementById("search-input").value.trim();
    if (inputVal) {
        triggerSearch();
    }
}

// ====================================================================
// Autocomplete Predictions
// ====================================================================
async function initAutocomplete() {
    try {
        const res = await fetch("/api/autocomplete");
        if (res.ok) {
            autocompleteData = await res.json();
        }
    } catch (e) {
        console.error("Autocomplete failure: ", e);
    }
}

function showAutocompleteSuggestions() {
    const input = document.getElementById("search-input");
    const query = input.value.trim().toLowerCase();
    const popup = document.getElementById("autocomplete-popup");
    
    if (!query) {
        popup.style.display = "none";
        return;
    }
    
    // Filter genes and drugs matching characters
    const matchingGenes = autocompleteData.genes.filter(g => g.toLowerCase().includes(query)).slice(0, 5);
    const matchingDrugs = autocompleteData.drugs.filter(d => d.toLowerCase().includes(query)).slice(0, 5);
    
    if (matchingGenes.length === 0 && matchingDrugs.length === 0) {
        popup.style.display = "none";
        return;
    }
    
    popup.innerHTML = "";
    
    matchingGenes.forEach(gene => {
        const item = document.createElement("div");
        item.className = "autocomplete-item";
        item.innerHTML = `<span class="autocomplete-badge badge-gene">gene</span> <strong>${gene}</strong>`;
        item.onclick = () => selectSuggestion(gene);
        popup.appendChild(item);
    });
    
    matchingDrugs.forEach(drug => {
        const item = document.createElement("div");
        item.className = "autocomplete-item";
        item.innerHTML = `<span class="autocomplete-badge badge-drug">drug</span> <strong>${drug}</strong>`;
        item.onclick = () => selectSuggestion(drug);
        popup.appendChild(item);
    });
    
    popup.style.display = "block";
}

function selectSuggestion(name) {
    document.getElementById("search-input").value = name;
    document.getElementById("autocomplete-popup").style.display = "none";
    triggerSearch();
}

// Close autocomplete popup on outer click
document.addEventListener("click", (e) => {
    if (e.target.id !== "search-input") {
        document.getElementById("autocomplete-popup").style.display = "none";
    }
});

// ====================================================================
// Search Query Calculations
// ====================================================================
async function triggerSearch() {
    const input = document.getElementById("search-input");
    const query = input.value.trim();
    
    if (!query) return;
    
    // Loading overlay trigger
    const loader = document.getElementById("graph-loader");
    loader.classList.add("active");
    
    try {
        // Determine query type (is it gene or drug)
        const isGene = autocompleteData.genes.some(g => g.toLowerCase() === query.toLowerCase());
        const isDrug = autocompleteData.drugs.some(d => d.toLowerCase() === query.toLowerCase());
        
        let url = "";
        let matchedName = query;
        
        if (isGene) {
            // Find casing matches
            matchedName = autocompleteData.genes.find(g => g.toLowerCase() === query.toLowerCase());
            url = `/api/recommend/gene?name=${matchedName}&method=${currentMethod}`;
        } else if (isDrug) {
            matchedName = autocompleteData.drugs.find(d => d.toLowerCase() === query.toLowerCase());
            url = `/api/recommend/drug?name=${matchedName}&method=${currentMethod}`;
        } else {
            // Fallback default search
            matchedName = query;
            url = `/api/recommend/gene?name=${matchedName}&method=${currentMethod}`;
        }
        
        input.value = matchedName; // Update text capitalization
        
        const recResponse = await fetch(url);
        if (!recResponse.ok) {
            const err = await recResponse.json();
            showEmptyState(err.detail || "Query Failed");
            loader.classList.remove("active");
            return;
        }
        
        const recData = await recResponse.json();
        
        // Populate results table
        populateTable(recData.results, isGene);
        
        // Fetch network structure
        const netResponse = await fetch(`/api/network?query=${matchedName}&method=${currentMethod}`);
        if (netResponse.ok) {
            const netData = await netResponse.json();
            initNetworkSimulation(netData, matchedName);
        }
    } catch (e) {
        showEmptyState("Connection error fetching recommendations.");
        console.error(e);
    } finally {
        loader.classList.remove("active");
    }
}

function populateTable(results, isGeneQuery) {
    const tableBody = document.getElementById("table-body");
    const badge = document.getElementById("results-count-badge");
    const headers = document.getElementById("table-headers");
    
    badge.textContent = `${results.length} Candidates Loaded`;
    
    // Set headers
    if (isGeneQuery) {
        headers.innerHTML = `
            <th>Rank</th>
            <th>Drug Candidate</th>
            <th>Embedding Match</th>
            <th>Association Type</th>
            <th>Key Indications</th>
        `;
    } else {
        headers.innerHTML = `
            <th>Rank</th>
            <th>Similar Drug</th>
            <th>Embedding Match</th>
            <th>Key Indications</th>
        `;
    }
    
    tableBody.innerHTML = "";
    
    if (results.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="5" class="empty-state">No candidates returned</td></tr>`;
        return;
    }
    
    results.forEach(row => {
        const tr = document.createElement("tr");
        const percentage = Math.round(row.score * 100);
        
        if (isGeneQuery) {
            const typeClass = row.type === "Direct Target" ? "type-direct" : "type-repurposed";
            tr.innerHTML = `
                <td style="font-weight: 700; color: #64748b;">#${row.rank}</td>
                <td style="font-weight: 600; color: #f0f3f8;">${row.drug}</td>
                <td>
                    <div class="score-cell">
                        <span class="score-num">${row.score.toFixed(3)}</span>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                </td>
                <td><span class="badge-type ${typeClass}">${row.type}</span></td>
                <td class="text-muted" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${row.indications}</td>
            `;
        } else {
            tr.innerHTML = `
                <td style="font-weight: 700; color: #64748b;">#${row.rank}</td>
                <td style="font-weight: 600; color: #f0f3f8;">${row.drug}</td>
                <td>
                    <div class="score-cell">
                        <span class="score-num">${row.score.toFixed(3)}</span>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: ${percentage}%"></div>
                        </div>
                    </div>
                </td>
                <td class="text-muted">${row.indications}</td>
            `;
        }
        
        tableBody.appendChild(tr);
    });
}

function showEmptyState(msg) {
    const tableBody = document.getElementById("table-body");
    tableBody.innerHTML = `
        <tr>
            <td colspan="5" class="empty-state">
                <i data-lucide="shield-alert"></i>
                <p>${msg}</p>
            </td>
        </tr>
    `;
    lucide.createIcons();
}

// ====================================================================
// Canvas Force-Directed Graph Physics Engine
// ====================================================================
function initNetworkSimulation(network, queryNodeName) {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Adapt canvas rendering sizes
    resizeCanvas();
    
    // Build Nodes and give them random coordinates
    graphData.nodes = network.nodes.map(n => {
        const isQuery = n.id === queryNodeName;
        return {
            ...n,
            x: canvas.width / 2 + (Math.random() - 0.5) * 100,
            y: canvas.height / 2 + (Math.random() - 0.5) * 100,
            vx: 0,
            vy: 0,
            radius: isQuery ? 15 : (n.type === "drug" ? 10 : 8),
            mass: isQuery ? 2.5 : 1.0,
            isQuery: isQuery
        };
    });
    
    // Map Links
    graphData.links = network.links;
    
    // Recenter
    scale = 1.0;
    offsetX = 0;
    offsetY = 0;
    
    // Run physics frame loop
    runPhysicsLoop();
}

function runPhysicsLoop() {
    updatePhysics();
    renderGraph();
    animationFrameId = requestAnimationFrame(runPhysicsLoop);
}

function updatePhysics() {
    const nodes = graphData.nodes;
    const links = graphData.links;
    
    // 1. Coulomb Node Repulsion Force (Nodes push each other away)
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const n1 = nodes[i];
            const n2 = nodes[j];
            
            const dx = n2.x - n1.x;
            const dy = n2.y - n1.y;
            const dist = Math.sqrt(dx * dx + dy * dy) || 1;
            
            // Force strength inversely proportional to distance squared
            const force = repulsionConstant / (dist * dist);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;
            
            n1.vx -= fx / n1.mass;
            n1.vy -= fy / n1.mass;
            n2.vx += fx / n2.mass;
            n2.vy += fy / n2.mass;
        }
    }
    
    // 2. Hooke Edge Attraction Force (Edges act as springs pulling connected nodes)
    links.forEach(link => {
        const sourceNode = nodes.find(n => n.id === link.source);
        const targetNode = nodes.find(n => n.id === link.target);
        
        if (!sourceNode || !targetNode) return;
        
        const dx = targetNode.x - sourceNode.x;
        const dy = targetNode.y - sourceNode.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        
        // Spring contraction force
        const displacement = dist - springLength;
        const force = springConstant * displacement;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        
        sourceNode.vx += fx / sourceNode.mass;
        sourceNode.vy += fy / sourceNode.mass;
        targetNode.vx -= fx / targetNode.mass;
        targetNode.vy -= fy / targetNode.mass;
    });
    
    // 3. Center Gravity Force (keeps the component unified in center)
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    nodes.forEach(node => {
        if (node === draggedNode) return;
        
        const dx = centerX - node.x;
        const dy = centerY - node.y;
        
        node.vx += dx * centerGravity;
        node.vy += dy * centerGravity;
        
        // Apply friction damping
        node.vx *= friction;
        node.vy *= friction;
        
        // Update positions
        node.x += node.vx;
        node.y += node.vy;
    });
}

function renderGraph() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.save();
    // Apply pan & zoom translation matrix
    ctx.translate(canvas.width / 2 + offsetX, canvas.height / 2 + offsetY);
    ctx.scale(scale, scale);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
    
    // 1. Draw Links
    graphData.links.forEach(link => {
        const s = graphData.nodes.find(n => n.id === link.source);
        const t = graphData.nodes.find(n => n.id === link.target);
        
        if (!s || !t) return;
        
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        
        // Link style matching biological mapping type
        if (link.type === "direct") {
            ctx.strokeStyle = "rgba(255, 87, 34, 0.4)";
            ctx.lineWidth = 2.5;
            ctx.setLineDash([]);
        } else if (link.type === "repurposed") {
            ctx.strokeStyle = "rgba(0, 240, 255, 0.35)";
            ctx.lineWidth = 1.8;
            ctx.setLineDash([4, 4]); // Dashed line representing implicit prediction
        } else if (link.type === "similarity") {
            ctx.strokeStyle = "rgba(138, 43, 226, 0.35)";
            ctx.lineWidth = 1.8;
            ctx.setLineDash([2, 4]);
        } else {
            ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
            ctx.lineWidth = 1.2;
            ctx.setLineDash([]);
        }
        ctx.stroke();
    });
    ctx.setLineDash([]); // Reset
    
    // 2. Draw Nodes
    graphData.nodes.forEach(node => {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
        
        let gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius);
        
        if (node.isQuery) {
            // Giant glowing query node
            ctx.shadowBlur = 20;
            ctx.shadowColor = node.type === "drug" ? "#00f0ff" : "#9d4edd";
            
            if (node.type === "drug") {
                gradient.addColorStop(0, "#00f0ff");
                gradient.addColorStop(1, "#00a8cc");
            } else {
                gradient.addColorStop(0, "#c77dff");
                gradient.addColorStop(1, "#7b2cbf");
            }
        } else {
            ctx.shadowBlur = 0;
            if (node.type === "drug") {
                gradient.addColorStop(0, "#00d2fc");
                gradient.addColorStop(1, "#0081a7");
            } else {
                gradient.addColorStop(0, "#9d4edd");
                gradient.addColorStop(1, "#5a189a");
            }
        }
        
        ctx.fillStyle = gradient;
        ctx.fill();
        
        // Highlight active hovered node
        if (node === hoveredNode) {
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 2.0;
            ctx.stroke();
        } else if (node.isQuery) {
            ctx.strokeStyle = "rgba(255,255,255,0.4)";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        } else {
            ctx.strokeStyle = "rgba(255,255,255,0.08)";
            ctx.lineWidth = 1.0;
            ctx.stroke();
        }
        
        // 3. Draw Labels
        ctx.shadowBlur = 0;
        ctx.fillStyle = node.isQuery ? "#ffffff" : "#cbd5e1";
        ctx.font = node.isQuery ? "bold 13px Inter" : "11px Inter";
        ctx.textAlign = "center";
        ctx.fillText(node.label, node.x, node.y - node.radius - 8);
    });
    
    ctx.restore();
}

function resetGraphPhysics() {
    offsetX = 0;
    offsetY = 0;
    scale = 1.0;
    
    // Scatter coordinates slightly to reboot movement
    graphData.nodes.forEach(node => {
        node.x = canvas.width / 2 + (Math.random() - 0.5) * 50;
        node.y = canvas.height / 2 + (Math.random() - 0.5) * 50;
        node.vx = 0;
        node.vy = 0;
    });
}

function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
}

// Canvas Mouse Controls (Dragging, Pan, Zoom)
canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Map canvas coordinates accounting for offset & scale
    const worldX = (clickX - canvas.width / 2 - offsetX) / scale + canvas.width / 2;
    const worldY = (clickY - canvas.height / 2 - offsetY) / scale + canvas.height / 2;
    
    // Check if clicked a node
    const clickedNode = graphData.nodes.find(node => {
        const dx = node.x - worldX;
        const dy = node.y - worldY;
        return Math.sqrt(dx * dx + dy * dy) <= node.radius + 3;
    });
    
    if (clickedNode) {
        draggedNode = clickedNode;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
    } else {
        isDraggingCanvas = true;
        startX = e.clientX - offsetX;
        startY = e.clientY - offsetY;
    }
});

canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const worldX = (mouseX - canvas.width / 2 - offsetX) / scale + canvas.width / 2;
    const worldY = (mouseY - canvas.height / 2 - offsetY) / scale + canvas.height / 2;
    
    // Check Node Hover status
    hoveredNode = graphData.nodes.find(node => {
        const dx = node.x - worldX;
        const dy = node.y - worldY;
        return Math.sqrt(dx * dx + dy * dy) <= node.radius + 3;
    });
    
    if (draggedNode) {
        draggedNode.x = worldX;
        draggedNode.y = worldY;
        draggedNode.vx = 0;
        draggedNode.vy = 0;
    } else if (isDraggingCanvas) {
        offsetX = e.clientX - startX;
        offsetY = e.clientY - startY;
    }
});

window.addEventListener("mouseup", () => {
    draggedNode = null;
    isDraggingCanvas = false;
});

// Zoom Wheel Control
canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = 1.1;
    if (e.deltaY < 0) {
        scale = Math.min(scale * zoomFactor, 3.0);
    } else {
        scale = Math.max(scale / zoomFactor, 0.4);
    }
});

// Resize Observer for Canvas Frame updates
new ResizeObserver(() => {
    if (graphData.nodes.length > 0) {
        resizeCanvas();
    }
}).observe(canvas.parentElement);

// ====================================================================
// Tab 2: Validation Dashboard UI & Chart.js Curves
// ====================================================================
async function loadValidationDashboard() {
    try {
        const res = await fetch("/api/validation");
        if (!res.ok) return;
        const metrics = await res.json();
        
        // 1. Populate Numeric Score Cards
        document.getElementById("val-svd-auroc").textContent = metrics.svd.auroc.toFixed(3);
        document.getElementById("val-svd-aupr").textContent = metrics.svd.aupr.toFixed(3);
        document.getElementById("val-svd-r10").textContent = `${(metrics.svd.recall_10 * 100).toFixed(1)}%`;
        document.getElementById("val-svd-r50").textContent = `${(metrics.svd.recall_50 * 100).toFixed(1)}%`;
        
        document.getElementById("val-gnn-auroc").textContent = metrics.gnn.auroc.toFixed(3);
        document.getElementById("val-gnn-aupr").textContent = metrics.gnn.aupr.toFixed(3);
        document.getElementById("val-gnn-r10").textContent = `${(metrics.gnn.recall_10 * 100).toFixed(1)}%`;
        document.getElementById("val-gnn-r50").textContent = `${(metrics.gnn.recall_50 * 100).toFixed(1)}%`;
        
        // 2. Render ROC Curves Chart
        renderROCChart(metrics.svd, metrics.gnn);
        
        // 3. Render PR Curves Chart
        renderPRChart(metrics.svd, metrics.gnn);
        
    } catch (e) {
        console.error("Dashboard render failed: ", e);
    }
}

function renderROCChart(svd, gnn) {
    const ctxRoc = document.getElementById("roc-chart").getContext("2d");
    
    if (rocChart) {
        rocChart.destroy();
    }
    
    // Draw diagonal baseline coordinates
    const diagonal = Array.from({length: 50}, (_, i) => ({x: i/49, y: i/49}));
    
    // Map ROC points
    const svdPoints = svd.fpr.map((fpr, i) => ({x: fpr, y: svd.tpr[i]}));
    const gnnPoints = gnn.fpr.map((fpr, i) => ({x: fpr, y: gnn.tpr[i]}));
    
    rocChart = new Chart(ctxRoc, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `GNN Autoencoder (AUC = ${gnn.auroc.toFixed(3)})`,
                    data: gnnPoints,
                    borderColor: '#39ff14',
                    borderWidth: 2.5,
                    backgroundColor: 'rgba(57, 255, 20, 0.04)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: `SVD Matrix Proximity (AUC = ${svd.auroc.toFixed(3)})`,
                    data: svdPoints,
                    borderColor: '#9d4edd',
                    borderWidth: 2,
                    backgroundColor: 'rgba(157, 78, 221, 0.02)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Random Baseline (AUC = 0.500)',
                    data: diagonal,
                    borderColor: '#64748b',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: getChartOptions('False Positive Rate', 'True Positive Rate')
    });
}

function renderPRChart(svd, gnn) {
    const ctxPr = document.getElementById("pr-chart").getContext("2d");
    
    if (prChart) {
        prChart.destroy();
    }
    
    // Map PR points
    const svdPoints = svd.recall.map((r, i) => ({x: r, y: svd.precision[i]}));
    const gnnPoints = gnn.recall.map((r, i) => ({x: r, y: gnn.precision[i]}));
    
    prChart = new Chart(ctxPr, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `GNN Autoencoder (AUPR = ${gnn.aupr.toFixed(3)})`,
                    data: gnnPoints,
                    borderColor: '#39ff14',
                    borderWidth: 2.5,
                    backgroundColor: 'rgba(57, 255, 20, 0.04)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: `SVD Matrix Proximity (AUPR = ${svd.aupr.toFixed(3)})`,
                    data: svdPoints,
                    borderColor: '#9d4edd',
                    borderWidth: 2,
                    backgroundColor: 'rgba(157, 78, 221, 0.02)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }
            ]
        },
        options: getChartOptions('Recall', 'Precision')
    });
}

function getChartOptions(xAxisTitle, yAxisTitle) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'linear',
                title: { display: true, text: xAxisTitle, color: '#94a3b8', font: { family: 'Inter', weight: 600 } },
                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                ticks: { color: '#64748b' },
                min: 0,
                max: 1
            },
            y: {
                type: 'linear',
                title: { display: true, text: yAxisTitle, color: '#94a3b8', font: { family: 'Inter', weight: 600 } },
                grid: { color: 'rgba(255, 255, 255, 0.03)' },
                ticks: { color: '#64748b' },
                min: 0,
                max: 1
            }
        },
        plugins: {
            legend: {
                labels: { color: '#f0f3f8', font: { family: 'Inter', size: 11 } }
            }
        }
    };
}

// ====================================================================
// System Initialization
// ====================================================================
window.onload = async () => {
    await initAutocomplete();
    
    // Trigger default search on load to present a beautiful initial visual
    const defaultSearch = "EGFR";
    document.getElementById("search-input").value = defaultSearch;
    triggerSearch();
};
