#!/bin/bash
# Performance monitoring script for macOS
# Run this while using the WebGL UI to monitor system impact

echo "🔍 MacOS Performance Monitor for WebGL UI"
echo "=========================================="
echo "Run this script in a separate terminal while using the app"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to get GPU usage (requires powermetrics, needs sudo)
get_gpu_usage() {
    if command -v powermetrics >/dev/null 2>&1; then
        # This requires sudo, so we'll show a simpler approach
        echo "GPU: Use 'sudo powermetrics -n 1 -f plist | grep -A1 gpu_energy_nj' for detailed GPU metrics"
    else
        echo "GPU: powermetrics not available"
    fi
}

# Function to monitor CPU and memory
monitor_resources() {
    while true; do
        # Clear screen for live updates
        clear
        
        echo "🔍 MacOS Performance Monitor - $(date)"
        echo "=========================================="
        
        # CPU Temperature (if available)
        if command -v osx-cpu-temp >/dev/null 2>&1; then
            temp=$(osx-cpu-temp)
            echo "🌡️  CPU Temperature: $temp"
        else
            echo "🌡️  CPU Temperature: Install 'osx-cpu-temp' for monitoring"
        fi
        
        # CPU Usage
        cpu_usage=$(ps -A -o %cpu | awk '{sum+=$1} END {printf "%.1f", sum}')
        echo "⚡ CPU Usage: ${cpu_usage}%"
        
        # Memory Usage
        memory_pressure=$(memory_pressure | head -1)
        echo "💾 $memory_pressure"
        
        # Memory details
        memory_stats=$(vm_stat | head -5 | tail -4 | awk '
        /Pages free/ { free = $3 }
        /Pages active/ { active = $3 }
        /Pages inactive/ { inactive = $3 }
        /Pages wired down/ { wired = $4 }
        END { 
            total = (free + active + inactive + wired) * 4096 / 1024 / 1024 / 1024
            used = (active + inactive + wired) * 4096 / 1024 / 1024 / 1024
            printf "💽 Memory: %.1fGB used / %.1fGB total", used, total
        }')
        echo "   $memory_stats"
        
        # Browser processes (Chrome, Safari, Firefox)
        echo ""
        echo "🌐 Browser Process Usage:"
        ps aux | grep -E "(Chrome|Safari|firefox)" | grep -v grep | head -5 | awk '{printf "   %-15s CPU: %5s%% MEM: %5s%%\n", $11, $3, $4}'
        
        # Fan speed (if available)
        if command -v istats >/dev/null 2>&1; then
            echo ""
            echo "🌪️  Fan Status:"
            istats fan
        fi
        
        echo ""
        echo "📊 Quick Health Check:"
        
        # Warnings based on thresholds
        if (( $(echo "$cpu_usage > 80" | bc -l) )); then
            echo "   ⚠️  HIGH CPU USAGE (${cpu_usage}%) - WebGL may be stressing your Mac"
        elif (( $(echo "$cpu_usage > 50" | bc -l) )); then
            echo "   🟡 MODERATE CPU USAGE (${cpu_usage}%) - Normal for WebGL"
        else
            echo "   ✅ LOW CPU USAGE (${cpu_usage}%) - System running smoothly"
        fi
        
        # Memory pressure check
        if echo "$memory_pressure" | grep -q "warn\|critical"; then
            echo "   ⚠️  MEMORY PRESSURE DETECTED - Consider closing other apps"
        else
            echo "   ✅ MEMORY PRESSURE: Normal"
        fi
        
        echo ""
        echo "💡 Tips:"
        echo "   • Use 'Low Power' mode in the UI if CPU usage stays >80%"
        echo "   • Toggle 'Perf' button in the UI to see real-time FPS"
        echo "   • Normal WebGL usage: 30-60% CPU, 60+ FPS"
        echo "   • Press Ctrl+C to stop monitoring"
        echo ""
        
        sleep 2
    done
}

# Check if bc is available for calculations
if ! command -v bc >/dev/null 2>&1; then
    echo "Installing bc for calculations..."
    # For homebrew users
    if command -v brew >/dev/null 2>&1; then
        brew install bc
    else
        echo "Please install 'bc' for accurate calculations: brew install bc"
    fi
fi

# Start monitoring
monitor_resources