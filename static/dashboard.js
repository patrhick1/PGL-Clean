/**
 * PGL Automation Dashboard JavaScript
 * Handles interactions with the automation API endpoints
 */

// Task Management Functions

// Store the last triggered task ID
let lastTriggeredTaskId = null;

async function triggerAutomation(action, id = null) {
    try {
        showStatusMessage('Starting automation...', 'info');
        
        const url = new URL('/trigger_automation', window.location.origin);
        url.searchParams.append('action', action);
        if (id) {
            url.searchParams.append('id', id);
        }
        
        const response = await fetch(url.toString());
        const data = await response.json();
        
        if (response.ok) {
            lastTriggeredTaskId = data.task_id;
            showStatusMessage(`Automation started successfully. Task ID: ${data.task_id}`, 'success');
            // Auto-refresh the task list
            listRunningTasks();
            // Auto-populate the task ID fields
            document.getElementById('stop-task-id').value = data.task_id;
            document.getElementById('status-task-id').value = data.task_id;
        } else {
            showStatusMessage(`Error: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        showStatusMessage(`Error: ${error.message}`, 'error');
    }
}

async function stopTask(taskId = null) {
    // If taskId is not provided, get it from the input field
    if (!taskId) {
        taskId = document.getElementById('stop-task-id').value.trim();
    }
    
    if (!taskId) {
        showStatusMessage('Please enter a task ID', 'error');
        return;
    }

    try {
        const response = await fetch(`/stop_task/${taskId}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (response.ok) {
            showStatusMessage(`${data.message}`, 'success');
            // Auto-refresh the task list
            listRunningTasks();
        } else {
            showStatusMessage(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatusMessage(`Error: ${error.message}`, 'error');
    }
}

async function checkTaskStatus(taskId = null) {
    // If taskId is not provided, get it from the input field
    if (!taskId) {
        taskId = document.getElementById('status-task-id').value.trim();
    }
    
    if (!taskId) {
        showStatusMessage('Please enter a task ID', 'error');
        return;
    }

    try {
        const response = await fetch(`/task_status/${taskId}`);
        const data = await response.json();

        const resultDisplay = document.getElementById('task-status-result');
        if (response.ok) {
            resultDisplay.innerHTML = `
                <strong>Task ID:</strong> ${data.task_id}<br>
                <strong>Action:</strong> ${data.action}<br>
                <strong>Status:</strong> ${data.status}<br>
                <strong>Runtime:</strong> ${data.runtime.toFixed(2)} seconds
            `;
            resultDisplay.classList.add('active');
        } else {
            resultDisplay.innerHTML = `Error: ${data.error}`;
            resultDisplay.classList.add('active');
        }
    } catch (error) {
        showStatusMessage(`Error: ${error.message}`, 'error');
    }
}

async function listRunningTasks() {
    try {
        const response = await fetch('/list_tasks');
        const data = await response.json();

        const tasksContainer = document.getElementById('running-tasks');
        
        if (response.ok) {
            if (Object.keys(data).length === 0) {
                tasksContainer.innerHTML = '<p>No running tasks</p>';
                return;
            }

            const tasksList = Object.values(data).map(task => `
                <div class="task-item">
                    <div class="task-info">
                        <span class="task-id">${task.task_id}</span>
                        <span class="task-status status-${task.status.toLowerCase()}">${task.status}</span>
                        <span class="task-runtime">${task.runtime.toFixed(2)}s</span>
                        <br>
                        <strong>Action:</strong> ${task.action}
                    </div>
                    <div class="task-actions">
                        <button class="btn-small" onclick="checkTaskStatus('${task.task_id}')">Status</button>
                        <button class="btn-small btn-danger" onclick="stopTask('${task.task_id}')">Stop</button>
                    </div>
                </div>
            `).join('');

            tasksContainer.innerHTML = tasksList;
        } else {
            tasksContainer.innerHTML = `<p>Error: ${data.error}</p>`;
        }
    } catch (error) {
        showStatusMessage(`Error: ${error.message}`, 'error');
    }
}

// Auto-populate task ID fields with the last triggered task ID
function populateLastTaskId(inputId) {
    if (lastTriggeredTaskId) {
        document.getElementById(inputId).value = lastTriggeredTaskId;
    }
}

// Modify the existing showStatusMessage function to support different message types
function showStatusMessage(message, type = 'info') {
    const statusDiv = document.getElementById('status-message');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;
    statusDiv.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 5000);
}

// Add event listeners when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initial task list load
    listRunningTasks();

    // Set up auto-refresh for task list (every 10 seconds)
    setInterval(listRunningTasks, 10000);
});

// Function to handle MIPR podcast search with ID input
async function handleMIPRSearch() {
    const id = prompt("Please enter the record ID:");
    if (id) {
        triggerAutomation('mipr_podcast_search', id);
    }
}

// Function to generate AI usage report
function generateAIUsageReport() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const groupBy = document.getElementById('group-by').value;
    const format = document.getElementById('format').value;
    
    let url = `/ai-usage?group_by=${groupBy}`;
    if (startDate) url += `&start_date=${startDate}`;
    if (endDate) url += `&end_date=${endDate}`;
    
    // Only open in new window if it's a format that returns data,
    // otherwise make a normal fetch request
    if (format === 'json' || format === 'text' || format === 'csv') {
        url += `&format=${format}`;
        window.open(url, '_blank');
    } else {
        fetch(url)
            .then(response => response.json())
            .then(data => {
                console.log('AI usage data:', data);
                alert('Report generated successfully');
            })
            .catch(error => alert("Error generating report: " + error));
    }
}

// Function to view podcast cost dashboard
function viewPodcastCostDashboard() {
    const podcastId = document.getElementById('podcast-id').value.trim();
    
    if (!podcastId) {
        alert("Please enter a valid Podcast Episode ID from the Podcast_Episodes table");
        return;
    }
    
    // Open dashboard in a new window
    window.open(`/podcast-cost-dashboard/${podcastId}`, '_blank');
}

// Function to transcribe a specific podcast episode
function transcribePodcast() {
    const podcastId = document.getElementById('transcribe-podcast-id').value.trim();
    
    if (!podcastId) {
        alert("Please enter a valid Podcast Episode ID from the Podcast_Episodes table");
        return;
    }
    
    // Confirm before starting transcription
    if (!confirm("This will start transcription for podcast episode ID: " + podcastId + ". Continue?")) {
        return;
    }
    
    // Show status message
    const statusDiv = document.getElementById('status-message');
    if (statusDiv) {
        statusDiv.textContent = `Starting transcription for episode ${podcastId}...`;
        statusDiv.className = 'status-running';
        statusDiv.style.display = 'block';
    }
    
    // Call the transcription endpoint
    fetch(`/transcribe-podcast/${podcastId}`, { method: 'GET' })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Transcription response:', data);
            if (statusDiv) {
                statusDiv.textContent = data.message || 'Transcription started!';
                statusDiv.className = 'status-success';
            }
            if (data.error) {
                alert("Error: " + data.error);
            } else {
                alert(data.message || "Transcription started successfully");
            }
        })
        .catch(error => {
            console.error('Transcription error:', error);
            if (statusDiv) {
                statusDiv.textContent = `Error: ${error.message}`;
                statusDiv.className = 'status-error';
            }
            alert("Error starting transcription: " + error.message);
        });
} 