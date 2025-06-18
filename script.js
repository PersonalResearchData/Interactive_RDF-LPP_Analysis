let uploadedFiles = [];
let processedData = null;

// File upload handling
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    uploadedFiles = Array.from(files).filter(file => file.name.endsWith('.xyz'));
    displayFileList();
    processBtn.disabled = uploadedFiles.length === 0;
}

function displayFileList() {
    fileList.innerHTML = '';
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span class="file-name">${file.name}</span>
            <span class="file-size">${(file.size / 1024).toFixed(1)} KB</span>
        `;
        fileList.appendChild(fileItem);
    });
}

// XYZ file parser
function parseXYZFile(content) {
    const lines = content.trim().split('\n');
    if (lines.length < 2) throw new Error('Invalid XYZ file format');

    const numParticles = parseInt(lines[0]);
    if (isNaN(numParticles)) throw new Error('Invalid particle count');

    const boxLine = lines[1].split(/\s+/);
    const boxIndex = boxLine.indexOf('box');
    if (boxIndex === -1 || boxLine.length < boxIndex + 7) {
        throw new Error('Invalid box information');
    }

    const [xlo, xhi, ylo, yhi, zlo, zhi] = boxLine.slice(boxIndex + 1, boxIndex + 7).map(parseFloat);
    const boxSize = [xhi - xlo, yhi - ylo, zhi - zlo];

    const positions = [];
    for (let i = 2; i < 2 + numParticles && i < lines.length; i++) {
        const parts = lines[i].split(/\s+/);
        if (parts.length >= 4) {
            positions.push([parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3])]);
        }
    }

    return { positions, boxSize };
}

// RDF calculation
function computeRDF(positions, boxSize, rMax, numBins) {
    const numParticles = positions.length;
    const rBin = Array.from({ length: numBins + 1 }, (_, i) => (i * rMax) / numBins);
    const dr = rBin[1] - rBin[0];
    const rdf = new Array(numBins).fill(0);

    const volume = boxSize[0] * boxSize[1] * boxSize[2];
    const density = numParticles / volume;

    for (let i = 0; i < numParticles; i++) {
        for (let j = 0; j < numParticles; j++) {
            if (i === j) continue;

            let dx = positions[i][0] - positions[j][0];
            let dy = positions[i][1] - positions[j][1];
            let dz = positions[i][2] - positions[j][2];

            // Periodic boundary conditions
            dx = dx - boxSize[0] * Math.round(dx / boxSize[0]);
            dy = dy - boxSize[1] * Math.round(dy / boxSize[1]);
            dz = dz - boxSize[2] * Math.round(dz / boxSize[2]);

            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (distance > 0 && distance < rMax) {
                const binIndex = Math.floor(distance / dr);
                if (binIndex < numBins) {
                    rdf[binIndex]++;
                }
            }
        }
    }

    // Normalize
    for (let i = 0; i < numBins; i++) {
        const r = rBin[i];
        const shellVolume = (4.0 / 3.0) * Math.PI * (Math.pow(r + dr, 3) - Math.pow(r, 3));
        rdf[i] = rdf[i] / (shellVolume * density * numParticles);
    }

    return { r: rBin.slice(0, -1), rdf };
}

// Euclidean distance function
function euclideanDistance(a, b) {
    return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
}

// Find k-nearest neighbors
function findKNearestNeighbors(data, k) {
    const n = data.length;
    const neighbors = [];
    const distances = [];

    for (let i = 0; i < n; i++) {
        const pointDistances = [];
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                const dist = euclideanDistance(data[i], data[j]);
                pointDistances.push({ index: j, distance: dist });
            }
        }
        
        // Sort by distance and take k nearest
        pointDistances.sort((a, b) => a.distance - b.distance);
        const kNearest = pointDistances.slice(0, k);
        
        neighbors.push(kNearest.map(neighbor => neighbor.index));
        distances.push(kNearest.map(neighbor => neighbor.distance));
    }

    return { neighbors, distances };
}

// Create weight matrix using heat kernel
function createWeightMatrix(data, neighbors, distances, sigma) {
    const n = data.length;
    const W = Array(n).fill().map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < neighbors[i].length; j++) {
            const neighborIndex = neighbors[i][j];
            const distance = distances[i][j];
            
            // Heat kernel: w_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            const weight = Math.exp(-Math.pow(distance, 2) / (2 * Math.pow(sigma, 2)));
            
            W[i][neighborIndex] = weight;
            W[neighborIndex][i] = weight; // Make symmetric
        }
    }

    return W;
}

// Create degree matrix
function createDegreeMatrix(W) {
    const n = W.length;
    const D = Array(n).fill().map(() => Array(n).fill(0));

    for (let i = 0; i < n; i++) {
        let degree = 0;
        for (let j = 0; j < n; j++) {
            degree += W[i][j];
        }
        D[i][i] = degree;
    }

    return D;
}

// Matrix operations
function matrixSubtract(A, B) {
    const result = [];
    for (let i = 0; i < A.length; i++) {
        result[i] = [];
        for (let j = 0; j < A[i].length; j++) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

function matrixMultiply(A, B) {
    const result = Array(A.length).fill().map(() => Array(B[0].length).fill(0));
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
            for (let k = 0; k < A[0].length; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

function transpose(matrix) {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
}

// Generalized eigenvalue problem solver using power iteration
function solveGeneralizedEigenvalue(A, B, nComponents) {
    const n = A.length;
    nComponents = Math.min(nComponents, n);
    
    const eigenPairs = [];
    const ACopy = A.map(row => [...row]);
    
    for (let comp = 0; comp < nComponents; comp++) {
        // Power iteration for smallest eigenvalue
        let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
        let lambda = 0;
        
        for (let iter = 0; iter < 200; iter++) {
            // Solve (A - lambda*B)v = 0, which is equivalent to A*v = lambda*B*v
            // For stability, we'll use A*v and normalize by B*v
            
            // v_new = A * v
            const Av = new Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Av[i] += ACopy[i][j] * v[j];
                }
            }
            
            // Bv = B * v
            const Bv = new Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Bv[i] += B[i][j] * v[j];
                }
            }
            
            // Rayleigh quotient: lambda = v^T A v / v^T B v
            const vAv = v.reduce((sum, val, i) => sum + val * Av[i], 0);
            const vBv = v.reduce((sum, val, i) => sum + val * Bv[i], 0);
            lambda = vAv / (vBv || 1e-10);
            
            // Normalize Av
            const norm = Math.sqrt(Av.reduce((sum, val) => sum + val * val, 0));
            if (norm > 1e-10) {
                v = Av.map(val => val / norm);
            }
        }
        
        eigenPairs.push({ value: lambda, vector: [...v] });
        
        // Deflate matrix
        const vBv = v.reduce((sum, val, i) => sum + val * B[i].reduce((s, Bval, j) => s + Bval * v[j], 0), 0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const Bv_i = B[i].reduce((s, Bval, k) => s + Bval * v[k], 0);
                const Bv_j = B[j].reduce((s, Bval, k) => s + Bval * v[k], 0);
                ACopy[i][j] -= (lambda * Bv_i * Bv_j) / (vBv || 1e-10);
            }
        }
    }
    
    // Sort by eigenvalue (ascending for LPP)
    eigenPairs.sort((a, b) => a.value - b.value);
    
    return eigenPairs;
}

// LPP implementation
function performLPP(data, k, sigma, nComponents) {
    const n = data.length;
    const m = data[0].length;
    nComponents = Math.min(nComponents, n, m);

    // Center the data
    const means = new Array(m).fill(0);
    for (let j = 0; j < m; j++) {
        for (let i = 0; i < n; i++) {
            means[j] += data[i][j];
        }
        means[j] /= n;
    }

    const centeredData = data.map(row => row.map((val, j) => val - means[j]));

    // Find k-nearest neighbors
    const { neighbors, distances } = findKNearestNeighbors(centeredData, k);

    // Create weight matrix W
    const W = createWeightMatrix(centeredData, neighbors, distances, sigma);

    // Create degree matrix D
    const D = createDegreeMatrix(W);

    // Compute Laplacian matrix L = D - W
    const L = matrixSubtract(D, W);

    // Compute data matrix X^T
    const X = centeredData;
    const Xt = transpose(X);

    // Compute A = X^T L X and B = X^T D X
    const XtL = matrixMultiply(Xt, L);
    const A = matrixMultiply(XtL, X);
    
    const XtD = matrixMultiply(Xt, D);
    const B = matrixMultiply(XtD, X);

    // Solve generalized eigenvalue problem A v = lambda B v
    const eigenPairs = solveGeneralizedEigenvalue(A, B, nComponents);

    // Project data onto LPP subspace
    const lppResult = centeredData.map(row => {
        const projected = [];
        for (let comp = 0; comp < nComponents; comp++) {
            const projection = row.reduce((sum, val, i) => sum + val * eigenPairs[comp].vector[i], 0);
            projected.push(projection);
        }
        return projected;
    });

    return {
        lppResult,
        eigenvalues: eigenPairs.map(p => p.value),
        eigenvectors: eigenPairs.map(p => p.vector),
        neighbors,
        distances,
        weightMatrix: W
    };
}

// Timestep extraction and label generation
function extractTimestep(filename) {
    const match = filename.match(/output_timestep_(\d+)(?:_.*)?\.xyz$/);
    return match ? parseInt(match[1]) : null;
}

function generateLabel(timestep, filename, timestepSize = 1.0) {
    if (timestep === null) return filename;
    
    const timePs = timestep * timestepSize / 1000; // Convert to ps
    
    if (timestep === 0) return "100 K, 0 fs";
    
    if (timePs < 1.0) {
        const timeFs = timestep * timestepSize;
        return `40 K, ${timeFs.toFixed(0)} fs`;
    } else if (timePs < 1000) {
        return `40 K, ${timePs.toFixed(2)} ps`;
    } else {
        const timeNs = timePs / 1000;
        return `40 K, ${timeNs.toFixed(2)} ns`;
    }
}

// Time averaging function
function performTimeAveraging(fileData, rdfs, labels, timestepSize, averagingWindow) {
    if (fileData.length === 0) return { fileData, rdfs, labels };

    // Convert averaging window from ps to timesteps
    const windowSteps = Math.round(averagingWindow / timestepSize * 1000);
    
    // Group files by time windows
    const timeGroups = new Map();
    
    fileData.forEach((item, index) => {
        const timestep = item.timestep || 0;
        const windowIndex = Math.floor(timestep / windowSteps);
        
        if (!timeGroups.has(windowIndex)) {
            timeGroups.set(windowIndex, []);
        }
        timeGroups.get(windowIndex).push({ index, timestep, item });
    });

    const averagedFileData = [];
    const averagedRdfs = [];
    const averagedLabels = [];

    // Average RDFs within each time window
    for (const [windowIndex, group] of timeGroups.entries()) {
        if (group.length === 0) continue;

        // Calculate average RDF for this window
        const avgRdf = new Array(rdfs[0].length).fill(0);
        let count = 0;

        group.forEach(({ index }) => {
            for (let i = 0; i < rdfs[index].length; i++) {
                avgRdf[i] += rdfs[index][i];
            }
            count++;
        });

        // Normalize by count
        for (let i = 0; i < avgRdf.length; i++) {
            avgRdf[i] /= count;
        }

        // Calculate representative timestep (center of window)
        const centerTimestep = (windowIndex + 0.5) * windowSteps;
        const timePs = centerTimestep * timestepSize / 1000;

        averagedFileData.push({
            timestep: centerTimestep,
            windowIndex,
            count,
            timeRange: group.map(g => g.timestep)
        });
        averagedRdfs.push(avgRdf);
        
        // Create label for averaged data
        const label = `40 K, ${timePs.toFixed(1)} ps (avg of ${count} files)`;
        averagedLabels.push(label);
    }

    return {
        fileData: averagedFileData,
        rdfs: averagedRdfs,
        labels: averagedLabels
    };
}

// Main processing function
async function processFiles() {
    if (uploadedFiles.length === 0) return;

    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        const rMax = parseFloat(document.getElementById('rMax').value);
        const numBins = parseInt(document.getElementById('numBins').value);
        const nComponents = parseInt(document.getElementById('lppComponents').value);
        const kNeighbors = parseInt(document.getElementById('kNeighbors').value);
        const sigma = parseFloat(document.getElementById('sigma').value);
        const timestepSize = parseFloat(document.getElementById('timestepSize').value);
        const averagingWindow = parseFloat(document.getElementById('averagingWindow').value);
        const enableAveraging = document.getElementById('enableAveraging').checked;

        const fileData = [];
        const rdfs = [];
        const labels = [];
        let rValues = null; // Store r values from first file

        // Process each file
        for (const file of uploadedFiles) {
            const content = await file.text();
            const { positions, boxSize } = parseXYZFile(content);
            const { r, rdf } = computeRDF(positions, boxSize, rMax, numBins);
            
            // Store r values from first file (all files should have same r values)
            if (rValues === null) {
                rValues = r;
            }
            
            const timestep = extractTimestep(file.name);
            const label = generateLabel(timestep, file.name, timestepSize);
            
            fileData.push({ file, timestep, positions, boxSize });
            rdfs.push(rdf);
            labels.push(label);
        }

        // Sort by timestep before processing
        const sortedIndices = fileData
            .map((item, index) => ({ index, timestep: item.timestep }))
            .sort((a, b) => (a.timestep || Infinity) - (b.timestep || Infinity))
            .map(item => item.index);

        let finalFileData = sortedIndices.map(i => fileData[i]);
        let finalRdfs = sortedIndices.map(i => rdfs[i]);
        let finalLabels = sortedIndices.map(i => labels[i]);

        // Apply time averaging if enabled
        if (enableAveraging) {
            const averaged = performTimeAveraging(finalFileData, finalRdfs, finalLabels, timestepSize, averagingWindow);
            finalFileData = averaged.fileData;
            finalRdfs = averaged.rdfs;
            finalLabels = averaged.labels;
        }

        // Standardize data for LPP
        const means = new Array(finalRdfs[0].length).fill(0);
        const stds = new Array(finalRdfs[0].length).fill(0);

        for (let j = 0; j < finalRdfs[0].length; j++) {
            for (let i = 0; i < finalRdfs.length; i++) {
                means[j] += finalRdfs[i][j];
            }
            means[j] /= finalRdfs.length;
        }

        for (let j = 0; j < finalRdfs[0].length; j++) {
            for (let i = 0; i < finalRdfs.length; i++) {
                stds[j] += Math.pow(finalRdfs[i][j] - means[j], 2);
            }
            stds[j] = Math.sqrt(stds[j] / (finalRdfs.length - 1));
        }

        const standardizedRdfs = finalRdfs.map(rdf => 
            rdf.map((val, j) => (val - means[j]) / (stds[j] || 1))
        );

        // Perform LPP
        const lppResults = performLPP(standardizedRdfs, kNeighbors, sigma, nComponents);

        processedData = {
            r: rValues, // Use the stored r values
            rdfs: finalRdfs,
            labels: finalLabels,
            lpp: lppResults,
            numFiles: uploadedFiles.length,
            numBins: numBins,
            processedFiles: finalRdfs.length,
            averagingEnabled: enableAveraging,
            averagingWindow: averagingWindow,
            timestepSize: timestepSize,
            kNeighbors: kNeighbors,
            sigma: sigma
        };

        displayResults();

    } catch (error) {
        console.error('Processing error:', error);
        alert('Error processing files: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function displayResults() {
    if (!processedData) return;

    document.getElementById('results').style.display = 'block';

    // Initialize section toggles after results are displayed
    setTimeout(() => {
        initializeSectionToggles();
    }, 100);

    // Display statistics
    displayStats();

    // Plot RDFs
    plotRDFs();

    // Plot LPP results
    plotLPP();

    // Plot eigenvalues
    plotEigenvalues();

    // Plot nearest neighbors graph
    plotNearestNeighbors();
}

function displayStats() {
    const statsGrid = document.getElementById('statsGrid');
    
    let statsHTML = `
        <div class="stat-card">
            <h3>${processedData.numFiles}</h3>
            <p>Original Files</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.processedFiles}</h3>
            <p>Processed Points</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.numBins}</h3>
            <p>RDF Bins</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.kNeighbors}</h3>
            <p>k-Neighbors</p>
        </div>
        <div class="stat-card">
            <h3>${processedData.sigma.toFixed(2)}</h3>
            <p>Heat Kernel σ</p>
        </div>
    `;

    if (processedData.averagingEnabled) {
        statsHTML += `
            <div class="stat-card">
                <h3>${processedData.averagingWindow}</h3>
                <p>Averaging Window (ps)</p>
            </div>
            <div class="stat-card">
                <h3>${processedData.timestepSize}</h3>
                <p>Timestep Size (ps)</p>
            </div>
        `;
    }

    statsGrid.innerHTML = statsHTML;
}

function plotRDFs() {
    const traces = processedData.rdfs.map((rdf, i) => ({
        x: processedData.r,
        y: rdf,
        type: 'scatter',
        mode: 'lines',
        name: processedData.labels[i],
        line: { width: 2 }
    }));

    const layout = {
        xaxis: { title: 'Distance (r)' },
        yaxis: { title: 'g(r)' },
        showlegend: true,
        legend: { x: 1.02, y: 1 },
        margin: { r: 150 }
    };

    Plotly.newPlot('rdfPlot', traces, layout, { responsive: true });
}

function plotLPP() {
    // Extract timestep values for color mapping
    let colorValues = [];
    let colorBarTitle = 'File Index';
    
    if (processedData.averagingEnabled) {
        // For averaged data, use the center timestep of each window
        colorValues = processedData.labels.map((label, index) => {
            // Extract time from label like "40 K, 10.5 ps (avg of 3 files)"
            const timeMatch = label.match(/(\d+(?:\.\d+)?)\s*(fs|ps|ns)/);
            if (timeMatch) {
                let timeValue = parseFloat(timeMatch[1]);
                const unit = timeMatch[2];
                
                // Convert everything to ps for consistent scaling
                if (unit === 'fs') timeValue /= 1000;
                else if (unit === 'ns') timeValue *= 1000;
                
                return timeValue;
            }
            return index; // Fallback to index
        });
        colorBarTitle = 'Time (ps)';
    } else {
        // For individual files, extract timesteps from original file processing
        colorValues = processedData.labels.map((label, index) => {
            // Try to extract timestep from the original file processing
            // Look for patterns like "100 K, 0 fs" or "40 K, 1.50 ps"
            const timeMatch = label.match(/(\d+(?:\.\d+)?)\s*(fs|ps|ns)/);
            if (timeMatch) {
                let timeValue = parseFloat(timeMatch[1]);
                const unit = timeMatch[2];
                
                // Convert everything to ps for consistent scaling
                if (unit === 'fs') timeValue /= 1000;
                else if (unit === 'ns') timeValue *= 1000;
                
                return timeValue;
            }
            return index; // Fallback to index
        });
        
        // Determine appropriate unit based on the range of values
        const maxTime = Math.max(...colorValues);
        if (maxTime < 1) {
            colorBarTitle = 'Time (fs)';
            colorValues = colorValues.map(t => t * 1000); // Convert back to fs
        } else if (maxTime < 1000) {
            colorBarTitle = 'Time (ps)';
        } else {
            colorBarTitle = 'Time (ns)';
            colorValues = colorValues.map(t => t / 1000); // Convert to ns
        }
    }

    const trace = {
        x: processedData.lpp.lppResult.map(point => point[0]),
        y: processedData.lpp.lppResult.map(point => point[1]),
        mode: 'markers+text',
        type: 'scatter',
        text: processedData.labels.map((label, i) => `${i+1}`), // Show point numbers
        textposition: 'top center',
        textfont: {
            size: 10,
            color: 'white'
        },
        marker: {
            size: 16,
            color: colorValues,
            colorscale: [
                [0, '#440154'],    // Dark purple (early times)
                [0.2, '#31688e'], // Dark blue
                [0.4, '#35b779'], // Green
                [0.6, '#fde725'], // Yellow
                [0.8, '#fd9731'], // Orange
                [1, '#cc4778']    // Pink-red (late times)
            ],
            colorbar: { 
                title: {
                    text: colorBarTitle,
                    font: { size: 14 }
                },
                thickness: 15,
                len: 0.7
            },
            line: {
                width: 2,
                color: 'white'
            }
        },
        hovertemplate: '<b>%{customdata}</b><br>' +
                      'LPP1: %{x:.3f}<br>' +
                      'LPP2: %{y:.3f}<br>' +
                      colorBarTitle + ': %{marker.color:.2f}<br>' +
                      '<b>Click to view RDF</b><br>' +
                      '<extra></extra>',
        customdata: processedData.labels
    };

    const layout = {
        xaxis: { 
            title: 'LPP Component 1',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'LPP Component 2',
            gridcolor: '#f0f0f0'
        },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        font: { family: 'Arial, sans-serif' },
        title: {
            text: 'LPP preserves local neighborhood structure - Click on any point to view its RDF',
            font: { size: 14, color: '#666' },
            x: 0.5
        }
    };

    const config = { 
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['select2d', 'lasso2d']
    };

    Plotly.newPlot('lppPlot', [trace], layout, config);

    // Add click event listener
    document.getElementById('lppPlot').on('plotly_click', function(data) {
        if (data.points && data.points.length > 0) {
            const pointIndex = data.points[0].pointIndex;
            displaySelectedRDF(pointIndex);
        }
    });
}

function displaySelectedRDF(selectedIndex) {
    if (!processedData || selectedIndex >= processedData.rdfs.length) return;

    // Show the selected RDF section if it's hidden
    const selectedRdfSection = document.getElementById('selected-rdf-section');
    const selectedRdfToggle = document.querySelector('[data-section="selected-rdf"]');
    
    if (!selectedRdfSection.classList.contains('visible')) {
        selectedRdfToggle.classList.add('active');
        selectedRdfToggle.classList.remove('inactive');
        selectedRdfSection.classList.add('visible');
        selectedRdfSection.style.display = 'block';
    }

    const selectedRdf = processedData.rdfs[selectedIndex];
    const selectedLabel = processedData.labels[selectedIndex];

    // Update title
    document.getElementById('selectedRdfTitle').textContent = 
        `🔍 Selected RDF: ${selectedLabel}`;

    // Create trace for selected RDF
    const selectedTrace = {
        x: processedData.r,
        y: selectedRdf,
        type: 'scatter',
        mode: 'lines',
        name: selectedLabel,
        line: { 
            width: 3, 
            color: '#e74c3c' 
        }
    };

    // Create comparison trace (all other RDFs in light gray)
    const allOtherTraces = processedData.rdfs.map((rdf, index) => {
        if (index === selectedIndex) return null;
        return {
            x: processedData.r,
            y: rdf,
            type: 'scatter',
            mode: 'lines',
            name: processedData.labels[index],
            line: { 
                width: 1, 
                color: 'rgba(150, 150, 150, 0.3)' 
            },
            showlegend: false,
            hovertemplate: `<b>${processedData.labels[index]}</b><br>r: %{x:.3f}<br>g(r): %{y:.3f}<extra></extra>`
        };
    }).filter(trace => trace !== null);

    const traces = [...allOtherTraces, selectedTrace];

    const layout = {
        xaxis: { 
            title: 'Distance (r)',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'g(r)',
            gridcolor: '#f0f0f0'
        },
        showlegend: true,
        legend: { 
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            bgcolor: 'rgba(255,255,255,0.8)'
        },
        margin: { 
            l: 60, 
            r: 60, 
            t: 40, 
            b: 80 
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        hovermode: 'x unified'
    };

    const config = { 
        responsive: true,
        displayModeBar: true 
    };

    Plotly.newPlot('selectedRdfPlot', traces, layout, config);

    // Scroll to the selected RDF plot
    document.getElementById('selectedRdfPlot').scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

function plotEigenvalues() {
    const nComponents = Math.min(processedData.lpp.eigenvalues.length, 10);
    const components = Array.from({length: nComponents}, (_, i) => `λ${i+1}`);
    
    const trace = {
        x: components,
        y: processedData.lpp.eigenvalues.slice(0, nComponents),
        type: 'bar',
        marker: {
            color: components.map((_, i) => `hsl(${240 + i * 20}, 70%, 60%)`)
        },
        text: processedData.lpp.eigenvalues.slice(0, nComponents).map(v => v.toFixed(4)),
        textposition: 'auto'
    };

    const layout = {
        xaxis: { title: 'LPP Component' },
        yaxis: { title: 'Eigenvalue' },
        title: {
            text: 'LPP Eigenvalues (smaller values indicate better locality preservation)',
            font: { size: 14 }
        }
    };

    Plotly.newPlot('eigenvaluesPlot', [trace], layout, { responsive: true });
}

function plotNearestNeighbors() {
    if (!processedData.lpp.neighbors || !processedData.lpp.lppResult) return;

    const lppCoords = processedData.lpp.lppResult;
    const neighbors = processedData.lpp.neighbors;

    // Create edges for the neighbor graph
    const edges = [];
    for (let i = 0; i < neighbors.length; i++) {
        for (let j = 0; j < neighbors[i].length; j++) {
            const neighborIndex = neighbors[i][j];
            edges.push({
                x: [lppCoords[i][0], lppCoords[neighborIndex][0], null],
                y: [lppCoords[i][1], lppCoords[neighborIndex][1], null]
            });
        }
    }

    // Create edge traces
    const edgeTrace = {
        x: edges.flatMap(edge => edge.x),
        y: edges.flatMap(edge => edge.y),
        mode: 'lines',
        line: {
            width: 1,
            color: 'rgba(128, 128, 128, 0.3)'
        },
        showlegend: false,
        hoverinfo: 'none'
    };

    // Create node trace
    const nodeTrace = {
        x: lppCoords.map(point => point[0]),
        y: lppCoords.map(point => point[1]),
        mode: 'markers+text',
        type: 'scatter',
        text: processedData.labels.map((label, i) => `${i+1}`),
        textposition: 'middle center',
        textfont: {
            size: 8,
            color: 'white'
        },
        marker: {
            size: 12,
            color: '#667eea',
            line: {
                width: 2,
                color: 'white'
            }
        },
        hovertemplate: '<b>%{customdata}</b><br>' +
                      'LPP1: %{x:.3f}<br>' +
                      'LPP2: %{y:.3f}<br>' +
                      '<extra></extra>',
        customdata: processedData.labels,
        name: 'Data Points'
    };

    const layout = {
        xaxis: { 
            title: 'LPP Component 1',
            gridcolor: '#f0f0f0'
        },
        yaxis: { 
            title: 'LPP Component 2',
            gridcolor: '#f0f0f0'
        },
        hovermode: 'closest',
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        showlegend: false,
        title: {
            text: `k-Nearest Neighbor Graph (k=${processedData.kNeighbors})`,
            font: { size: 14, color: '#333' },
            x: 0.5
        }
    };

    const config = { 
        responsive: true,
        displayModeBar: true 
    };

    Plotly.newPlot('nearestNeighborsPlot', [edgeTrace, nodeTrace], layout, config);
}

// Section toggle functionality
function initializeSectionToggles() {
    const toggleButtons = document.querySelectorAll('.section-toggle');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sectionName = this.dataset.section;
            const section = document.getElementById(`${sectionName}-section`);
            
            if (!section) return;
            
            // Toggle button state
            if (this.classList.contains('active')) {
                this.classList.remove('active');
                this.classList.add('inactive');
                
                // Hide section
                section.classList.remove('visible');
                section.style.display = 'none';
            } else {
                this.classList.remove('inactive');
                this.classList.add('active');
                
                // Show section
                section.style.display = 'block';
                section.classList.add('visible');
                
                // Re-render the plot if it's being shown
                setTimeout(() => {
                    if (processedData) {
                        switch(sectionName) {
                            case 'rdf':
                                plotRDFs();
                                break;
                            case 'lpp':
                                plotLPP();
                                break;
                            case 'eigenvalues':
                                plotEigenvalues();
                                break;
                            case 'nearest-neighbors':
                                plotNearestNeighbors();
                                break;
                        }
                    }
                }, 100);
            }
        });
    });
}

// Initialize toggles when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Will be initialized when displayResults() is called
});
