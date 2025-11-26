import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, TYPE_CHECKING
from src.examples_gen.unit_load import UnitLoad

if TYPE_CHECKING:
    from src.bay.buffer import Buffer


def calculate_deadline_factor(deadline: int, current_time: int, planning_horizon: int) -> float:
    """Calculates urgency based on the deadline, normalized to [0, 1]."""
    if planning_horizon <= 0:
        return 1.0
    
    time_to_deadline = deadline - current_time
    factor = 1.0 - (time_to_deadline / planning_horizon)
    return max(0.0, min(1.0, factor))

def calculate_slack_factor(deadline: int, current_time: int, planning_horizon: int, estimated_task_time: int) -> float:
    """Calculates urgency based on slack time, normalized to [0, 1]."""
    # A simple normalization constant; can be tuned
    max_reasonable_slack = planning_horizon / 2.0
    if max_reasonable_slack <= 0:
        return 1.0

    time_to_deadline = deadline - current_time
    slack_time = time_to_deadline - estimated_task_time
    
    if slack_time <= 0:
        return 1.0  # Max urgency if slack is zero or negative
    
    factor = 1.0 - (slack_time / max_reasonable_slack)
    return max(0.0, min(1.0, factor))

def estimate_task_time(task, buffer_state=None, buffer=None):
    """
    Estimate the time needed to complete a task using the same efficient heuristic as h_cost.
    Optimized for performance while maintaining accuracy.
    
    Time Calculation Details:
    ========================
    
    Storage Tasks:
    - Empty lanes: base_time (direct storage)
    - Non-empty lanes: base_time + avg_reshuffling_time
    - No available lanes: base_time + avg_reshuffling_time (fallback)
    
    Retrieval Tasks:
    - Accessible items: base_time (direct retrieval)
    - Blocked items: base_time + (avg_reshuffling_time × num_blockers)
    - Missing items: high_penalty_time (invalid state)
    
    Args:
        task: The task object (storage or retrieval)
        buffer_state: Current state of the buffer/buffer (optional)
        buffer: Buffer object for blocking analysis (optional)
    
    Returns:
        int: Estimated time units needed to complete the task
    """
    # Time constants (matching h_cost approach)
    base_time = 2                    # Base time for any operation
    avg_reshuffling_time = 4         # Average time for one reshuffling move (2 × avg_distance / speed)
    high_penalty_time = 100          # Penalty for invalid/impossible tasks
    
    # Fast buffer-aware analysis (similar to h_cost)
    if buffer is not None:
        try:
            if hasattr(task, 'is_mock') and task.is_mock:
                # --- Storage Task Time Estimation (matching h_cost logic) ---
                
                # Quick check for empty lanes
                empty_lanes = buffer.get_all_empty_slots()
                if empty_lanes:
                    # Direct storage in empty lane - minimal time
                    return base_time
                else:
                    # Need reshuffling - use average estimate for performance
                    # (same logic as h_cost fallback)
                    return base_time + avg_reshuffling_time
                        
            else:
                # --- Retrieval Task Time Estimation (matching h_cost logic) ---
                
                ul_id = task.id
                num_blockers = buffer.get_number_of_blockers(ul_id)
                
                if num_blockers == 0:
                    # Item is accessible - direct retrieval
                    return base_time
                elif num_blockers == float('inf'):
                    # Item not found in buffer - invalid state
                    return high_penalty_time
                else:
                    # Item is blocked - time proportional to blockers
                    # (same logic as h_cost: avg_reshuffling_time × num_blockers)
                    return base_time + (avg_reshuffling_time * num_blockers)
                    
        except Exception:
            # Fast fallback to buffer_state analysis
            pass
    
    # --- Fast Buffer State Analysis (performance-optimized) ---
    if buffer_state is not None:
        # Quick fill level calculation
        fill_level = sum(1 for slot in buffer_state.values() if slot is not None) / max(len(buffer_state), 1)
        
        if hasattr(task, 'is_mock') and task.is_mock:
            # Storage task - time increases with fill level
            if fill_level < 0.7:
                return base_time  # Plenty of space available
            else:
                return base_time + avg_reshuffling_time  # Some reshuffling needed
        else:
            # Retrieval task - estimate blockers based on fill level
            if fill_level < 0.4:
                return base_time  # Low density - likely accessible
            elif fill_level < 0.8:
                # Medium density - estimate 1-2 blockers
                estimated_blockers = 1
                return base_time + (avg_reshuffling_time * estimated_blockers)
            else:
                # High density - estimate 2-3 blockers
                estimated_blockers = 2
                return base_time + (avg_reshuffling_time * estimated_blockers)
    
    # --- Ultra-fast Fallback (when no state information available) ---
    if hasattr(task, 'is_mock') and task.is_mock:
        # Storage task - assume minimal complexity
        return base_time + 1
    else:
        # Retrieval task - assume average complexity
        return base_time + avg_reshuffling_time

def calculate_task_urgency(
    unit_loads: List, 
    buffer_state: Dict[tuple, Any] = None,
    current_time: int = 0,
    planning_horizon: int = None,
    weights: Dict[str, float] = None,
    verbose: bool = False
) -> List:
    """
    Calculates a sophisticated urgency score for all tasks (storage and retrieval)
    and returns a sorted list of tasks from most to least urgent.

    """
    if weights is None:
        weights = {'deadline': 0.4, 'slack': 0.6}  # Default weights
    
    if buffer_state is None:
        buffer_state = {}  # Empty buffer state if not provided
    
    # Calculate planning horizon if not provided
    if planning_horizon is None:
        max_time = max(ul.retrieval_end for ul in unit_loads if ul.retrieval_end is not None)
        min_time = min(ul.arrival_start for ul in unit_loads if ul.arrival_start is not None and ul.arrival_start > 0)
        planning_horizon = max_time - current_time if max_time else 100

    task_list = []
    
    # Create a unified list of all storage and retrieval tasks
    for ul in unit_loads:
        # Add the original retrieval task
        if ul.retrieval_end is not None:
            # Create a copy to avoid modifying original objects
            retrieval_task = UnitLoad(
                id=ul.id,
                retrieval_start=ul.retrieval_start,
                retrieval_end=ul.retrieval_end,
                arrival_start=ul.arrival_start,
                arrival_end=ul.arrival_end
            )
            retrieval_task.is_mock = False
            retrieval_task.task_type = "RETRIEVAL"
            task_list.append(retrieval_task)
        
        # Add a "mock" storage task if the UL is not yet stored
        if not getattr(ul, 'stored', False):
            storage_task = UnitLoad(
                id=f"{ul.id}_mock",
                retrieval_start=ul.arrival_start,
                retrieval_end=ul.arrival_end,  # The "deadline" for storage is the end of its arrival window
                arrival_start=None,
                arrival_end=None
            )
            storage_task.is_mock = True
            storage_task.task_type = "STORAGE"
            storage_task.real_ul_id = ul.id
            task_list.append(storage_task)

    # --- Calculate Urgency for Each Task ---
    fill_level = len([slot for slot in buffer_state.values() if slot is not None]) / max(len(buffer_state), 1)
    
    # Dynamic weights based on fill level
    w_retrieve = 0.5 + fill_level
    w_store = 1.5 - fill_level
    
    for task in task_list:
        deadline = task.retrieval_end
        
        # 1. Estimate time needed for the task
        est_time = estimate_task_time(task, buffer_state)
        task.estimated_time = est_time
        
        # 2. Calculate Deadline and Slack Factors
        deadline_factor = calculate_deadline_factor(deadline, current_time, planning_horizon)
        slack_factor = calculate_slack_factor(deadline, current_time, planning_horizon, est_time)
        task.deadline_factor = deadline_factor
        task.slack_factor = slack_factor

        # 3. Calculate Base Urgency
        base_urgency = (weights['deadline'] * deadline_factor) + (weights['slack'] * slack_factor)
        
        # 4. Apply State-Dependent Multiplicative Weight
        if task.task_type == "RETRIEVAL":
            final_urgency = base_urgency * w_retrieve
        else:  # STORAGE
            final_urgency = base_urgency * w_store
            
        task.urgency = final_urgency
        
        # Calculate slack time for compatibility with existing code
        time_to_deadline = deadline - current_time
        task.slack_time = time_to_deadline - est_time

    # Sort tasks by final urgency score (descending)
    sorted_tasks = sorted(task_list, key=lambda x: x.urgency, reverse=True)
    
    # Assign integer priority for visualization and debugging
    priorities_dict = {}
    for i, task in enumerate(sorted_tasks):
        task.set_priority(i + 1)
        priorities_dict[task.id] = task.get_priority()
    
    if verbose:
        print("\n--- Task Urgency Calculation Summary ---")
        print(f"Current Time: {current_time}, Planning Horizon: {planning_horizon}, Fill Level: {fill_level:.2f}")
        print(f"Retrieval Weight: {w_retrieve:.2f}, Storage Weight: {w_store:.2f}")
        print("-" * 80)
        for task in sorted_tasks:
            print(f"P{task.get_priority():2d} | UL {task.id:<10} | Type: {task.task_type:<10} | Urgency: {task.urgency:.3f} "
                  f"| Factors (D:{task.deadline_factor:.2f}, S:{task.slack_factor:.2f})")
        print("-" * 80)

        # Calculate timeline parameters for visualization
        max_time = max(task.retrieval_end for task in sorted_tasks if task.retrieval_end is not None)
        min_time = 0
        total_runtime = max_time - min_time
        
        draw_timeline(sorted_tasks, priorities_dict, total_runtime, min_time, max_time)

    return sorted_tasks

def draw_timeline(sorted_uls, priorities, total_runtime, min_time, max_time):
    """
    Visualize the time windows of unit loads and their priorities.
    """
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, max(8, len(sorted_uls) * 0.8)))

    # Set up the plot
    ax.set_xlim(min_time - 5, max_time + 5)
    ax.set_ylim(-0.5, len(sorted_uls) - 0.5)
    
    colors = {
        'stored': 'lightblue',      # Already in buffer
        'arriving': 'orange',       # Needs to arrive first
        'retrieval': 'lightgreen',  # Retrieval window
        'deadline': 'red'           # Retrieval deadline
    }
    
    y_positions = []
    labels = []
    
    for i, ul in enumerate(sorted_uls):
        y_pos = len(sorted_uls) - 1 - i  # Reverse order for better readability
        y_positions.append(y_pos)
        
        # Create label with priority and slack time info
        priority = priorities[ul.id]
        slack_info = f", S:{ul.slack_time:.1f}" if hasattr(ul, 'slack_time') else ""
        label = f"UL {ul.id} (P{priority}{slack_info})"
        labels.append(label)
        
        # Draw total runtime bar (background)
        total_bar = patches.Rectangle((min_time, y_pos - 0.3), total_runtime, 0.6, 
                                    linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(total_bar)
        
        # Handle arrival window
        if ul.arrival_start is not None and ul.arrival_start > 0:
            # Unit load is arriving (orange)
            arrival_width = ul.arrival_end - ul.arrival_start if ul.arrival_end else 1
            arrival_bar = patches.Rectangle((ul.arrival_start, y_pos - 0.25), arrival_width, 0.5,
                                          linewidth=1, edgecolor='darkorange', facecolor=colors['arriving'], alpha=0.8)
            ax.add_patch(arrival_bar)
            
            # Add arrival window text
            ax.text(ul.arrival_start + arrival_width/2, y_pos, f"A[{ul.arrival_start}-{ul.arrival_end}]", 
                   ha='center', va='center', fontsize=8, weight='bold')
        else:
            # Unit load is already stored (blue)
            stored_bar = patches.Rectangle((min_time, y_pos - 0.25), max(10, total_runtime * 0.1), 0.5,
                                         linewidth=1, edgecolor='darkblue', facecolor=colors['stored'], alpha=0.8)
            ax.add_patch(stored_bar)
            
            # Add stored indicator
            ax.text(min_time + 5, y_pos, "STORED", ha='center', va='center', 
                   fontsize=8, weight='bold', color='darkblue')
        
        # Draw retrieval window (green)
        retrieval_width = ul.retrieval_end - ul.retrieval_start
        retrieval_bar = patches.Rectangle((ul.retrieval_start, y_pos - 0.2), retrieval_width, 0.4,
                                        linewidth=2, edgecolor='darkgreen', facecolor=colors['retrieval'], alpha=0.7)
        ax.add_patch(retrieval_bar)
        
        # Add retrieval window text
        ax.text(ul.retrieval_start + retrieval_width/2, y_pos, f"R[{ul.retrieval_start}-{ul.retrieval_end}]", 
               ha='center', va='center', fontsize=8, weight='bold')
        
        # Add deadline marker (red line)
        ax.axvline(x=ul.retrieval_end, ymin=(y_pos-0.4+0.5)/len(sorted_uls), ymax=(y_pos+0.4+0.5)/len(sorted_uls), 
                  color=colors['deadline'], linewidth=3, alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Unit Loads', fontsize=12)
    ax.set_title(f'Unit Load Time Windows and Priorities (Least Slack Time Scheduling)\n(Total Runtime: {total_runtime} time steps)', 
                fontsize=14, weight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Create legend
    legend_elements = [
        patches.Patch(color=colors['stored'], label='Already Stored'),
        patches.Patch(color=colors['arriving'], label='Arrival Window'),
        patches.Patch(color=colors['retrieval'], label='Retrieval Window'),
        patches.Rectangle((0,0),1,1, color=colors['deadline'], label='Retrieval Deadline')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save the plot
    plt.savefig('unit_load_timeline.png', dpi=300, bbox_inches='tight')
    print("Timeline visualization saved as 'unit_load_timeline.png'")
    
    # Show the plot
    plt.show()
    
    # Print priority summary
    print(f"\nPriority Summary (based on least slack time):")
    print("=" * 60)
    for ul in sorted_uls:
        priority = priorities[ul.id]
        status = "ARRIVING" if (ul.arrival_start is not None and ul.arrival_start > 0) else "STORED"
        arrival_info = f"arrives {ul.arrival_start}-{ul.arrival_end}" if status == "ARRIVING" else "already stored"
        slack_info = f"slack: {ul.slack_time:.1f}" if hasattr(ul, 'slack_time') else "slack: N/A"
        print(f"UL {ul.id}: Priority {priority:2d} | Retrieval: {ul.retrieval_start:3d}-{ul.retrieval_end:3d} | {arrival_info} | {slack_info}")


def draw_edd_timeline(all_tasks, priorities):
    """
    Visualize the EDD (Earliest Due Date) priority assignment on a timeline.
    """
    # Extract unique unit loads for visualization
    unique_uls = {}
    for task in all_tasks:
        ul = task['ul']
        if ul.id not in unique_uls:
            unique_uls[ul.id] = ul
    
    uls = list(unique_uls.values())
    
    # Calculate time bounds
    min_time = 0
    max_time = max(task['deadline'] for task in all_tasks)
    total_runtime = max_time - min_time
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(15, max(8, len(uls) * 0.8)))
    
    # Set up the plot
    ax.set_xlim(min_time - 5, max_time + 5)
    ax.set_ylim(-0.5, len(uls) - 0.5)
    
    colors = {
        'stored': 'lightblue',
        'arriving': 'orange', 
        'retrieval': 'lightgreen',
        'deadline': 'red'
    }
    
    y_positions = []
    labels = []
    
    # Sort unit loads by their EDD priority for display
    task_priorities = {}
    for task in all_tasks:
        if task['type'] == 'retrieval':
            task_priorities[task['ul'].id] = priorities[task['id']]
    
    sorted_uls = sorted(uls, key=lambda ul: task_priorities.get(ul.id, float('inf')))
    
    for i, ul in enumerate(sorted_uls):
        y_pos = len(sorted_uls) - 1 - i
        y_positions.append(y_pos)
        
        # Get EDD priority
        priority = task_priorities.get(ul.id, 'N/A')
        labels.append(f"UL {ul.id} (EDD Priority: {priority})")
        
        # Draw background
        total_bar = patches.Rectangle((min_time, y_pos - 0.3), total_runtime, 0.6,
                                    linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(total_bar)
        
        # Handle arrival window
        if ul.arrival_start is not None and ul.arrival_start > 0:
            arrival_width = ul.arrival_end - ul.arrival_start if ul.arrival_end else 1
            arrival_bar = patches.Rectangle((ul.arrival_start, y_pos - 0.25), arrival_width, 0.5,
                                          linewidth=1, edgecolor='darkorange', facecolor=colors['arriving'], alpha=0.8)
            ax.add_patch(arrival_bar)
            ax.text(ul.arrival_start + arrival_width/2, y_pos, f"A[{ul.arrival_start}-{ul.arrival_end}]",
                   ha='center', va='center', fontsize=8, weight='bold')
        else:
            # Already stored
            stored_bar = patches.Rectangle((min_time, y_pos - 0.25), max(10, total_runtime * 0.1), 0.5,
                                         linewidth=1, edgecolor='darkblue', facecolor=colors['stored'], alpha=0.8)
            ax.add_patch(stored_bar)
            ax.text(min_time + 5, y_pos, "STORED", ha='center', va='center',
                   fontsize=8, weight='bold', color='darkblue')
        
        # Draw retrieval window
        if ul.retrieval_start is not None and ul.retrieval_end is not None:
            retrieval_width = ul.retrieval_end - ul.retrieval_start
            retrieval_bar = patches.Rectangle((ul.retrieval_start, y_pos - 0.2), retrieval_width, 0.4,
                                            linewidth=2, edgecolor='darkgreen', facecolor=colors['retrieval'], alpha=0.7)
            ax.add_patch(retrieval_bar)
            ax.text(ul.retrieval_start + retrieval_width/2, y_pos, f"R[{ul.retrieval_start}-{ul.retrieval_end}]",
                   ha='center', va='center', fontsize=8, weight='bold')
            
            # Add deadline marker
            ax.axvline(x=ul.retrieval_end, ymin=(y_pos-0.4+0.5)/len(sorted_uls), 
                      ymax=(y_pos+0.4+0.5)/len(sorted_uls),
                      color=colors['deadline'], linewidth=3, alpha=0.8)
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Unit Loads', fontsize=12)
    ax.set_title('Earliest Due Date (EDD) Priority Assignment\n(Earlier deadline = Higher priority)', 
                fontsize=14, weight='bold')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    
    # Create legend
    legend_elements = [
        patches.Patch(color=colors['stored'], label='Already Stored'),
        patches.Patch(color=colors['arriving'], label='Arrival Window'),
        patches.Patch(color=colors['retrieval'], label='Retrieval Window'),
        patches.Rectangle((0,0),1,1, color=colors['deadline'], label='Retrieval Deadline (EDD Sort Key)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save the plot
    plt.savefig('edd_timeline.png', dpi=300, bbox_inches='tight')
    print("EDD timeline visualization saved as 'edd_timeline.png'")
    
    # Show the plot
    plt.show()


def tws_to_priorities(uls, verbose=False): 
    """
    Create a dictionary mapping unit load IDs to their priorities using Earliest Due Date (EDD) algorithm.
    EDD prioritizes tasks based on their deadlines - earliest deadline gets highest priority.
    """
    if not uls:
        print("No unit loads found to prioritize.")
        return {}

    # Create list of all tasks with their deadlines
    all_tasks = []
    
    for ul in uls:
        # Add retrieval task
        if ul.retrieval_end is not None:
            all_tasks.append({
                'id': ul.id,
                'type': 'retrieval',
                'deadline': ul.retrieval_end,
                'ul': ul
            })
        
        # Add storage task (arrival deadline) if unit load needs to arrive
        if ul.arrival_start is not None and ul.arrival_start > 0:
            all_tasks.append({
                'id': f"{ul.id}_mock",
                'type': 'storage', 
                'deadline': ul.arrival_end,
                'ul': ul
            })
    
    # Sort by Earliest Due Date (EDD) - earliest deadline gets priority 1
    all_tasks.sort(key=lambda task: task['deadline'])
    
    # Create priority dictionary
    priorities = {}
    for i, task in enumerate(all_tasks):
        priorities[task['id']] = i + 1  # Priority 1 = highest priority (earliest deadline)
        task['ul'].set_priority(i + 1)
    
    if verbose:
        print("\n--- Earliest Due Date (EDD) Priority Assignment ---")
        print(f"Total tasks: {len(all_tasks)}")
        print("-" * 60)
        for i, task in enumerate(all_tasks):
            print(f"Priority {i+1:2d}: UL {task['id']:<12} | Type: {task['type']:<9} | Deadline: {task['deadline']:3d}")
        print("-" * 60)
        
        # Draw timeline with EDD priorities
        if len(uls) > 0:
            draw_edd_timeline(all_tasks, priorities)
    
    return priorities

def draw_task_queue_timeline(sorted_tasks: List[UnitLoad]):
    """
    Visualize the timeline for tasks sorted by the create_task_queue logic.
    """
    if not sorted_tasks:
        print("No tasks to visualize.")
        return

    # --- Calculate time bounds ---
    min_time = 0
    max_time = 0
    for task in sorted_tasks:
        if task.deadline is not None and task.deadline > max_time:
            max_time = task.deadline
    total_runtime = max_time - min_time
    
    # --- Create visualization ---
    fig, ax = plt.subplots(figsize=(15, max(8, len(sorted_tasks) * 0.6)))
    
    # --- Set up the plot ---
    ax.set_xlim(min_time - 5, max_time + 5)
    ax.set_ylim(-0.5, len(sorted_tasks) - 0.5)
    
    colors = {
        'STORAGE': 'orange',
        'RETRIEVAL': 'lightgreen',
        'deadline': 'red'
    }
    
    y_positions = []
    labels = []
    
    # --- Plot each task ---
    # Reverse order for top-to-bottom display of high priority
    for i, task in enumerate(reversed(sorted_tasks)):
        y_pos = i 
        y_positions.append(y_pos)
        
        # Label with ID and Group Priority
        label = f"Task {task.id} (P: {task.get_priority()})"
        labels.append(label)
        
        # Draw background bar
        total_bar = patches.Rectangle((min_time, y_pos - 0.3), total_runtime, 0.6,
                                      linewidth=1, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(total_bar)
        
        # Draw the task's time window
        tw_start = task.tw_start
        tw_width = task.deadline - tw_start
        task_color = colors.get(task.task_type, 'cyan')
        
        tw_bar = patches.Rectangle((tw_start, y_pos - 0.25), tw_width, 0.5,
                                   linewidth=1, edgecolor='black', facecolor=task_color, alpha=0.8)
        ax.add_patch(tw_bar)
        
        # Add time window text
        ax.text(tw_start + tw_width / 2, y_pos, f"TW [{tw_start}-{task.deadline}]",
                ha='center', va='center', fontsize=9, weight='bold')
        
        # Add deadline marker
        ax.axvline(x=task.deadline, ymin=(y_pos - 0.4 + 0.5) / len(sorted_tasks),
                   ymax=(y_pos + 0.4 + 0.5) / len(sorted_tasks),
                   color=colors['deadline'], linewidth=3, alpha=0.8)

    # --- Customize the plot ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Tasks Sorted by Priority', fontsize=12)
    ax.set_title('Task Queue Timeline (Enhanced EDD with Grouping)',
                 fontsize=14, weight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # --- Create legend ---
    legend_elements = [
        patches.Patch(color=colors['STORAGE'], label='Storage Task Window'),
        patches.Patch(color=colors['RETRIEVAL'], label='Retrieval Task Window'),
        patches.Rectangle((0, 0), 1, 1, color=colors['deadline'], label='Task Deadline')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    plt.savefig('task_queue_timeline.png', dpi=300, bbox_inches='tight')
    print("\nTimeline visualization saved as 'task_queue_timeline.png'")
    plt.show()

def create_task_queue(uls, buffer_state=None, current_time=0, verbose=False, draw_prio_image=False):
    """
    Creates a chronological, sorted list of all tasks (retrievals and storage)
    using an enhanced Earliest Due Date (EDD) algorithm that groups tasks with
    enclosed time windows.
    
    This is Step 1 of the heuristic pipeline.

    Args:
        uls (list): The list of UnitLoad objects from the instance.
        buffer_state (dict): Current state of the buffer/buffer (optional).
        current_time (int): Current time in the simulation (default: 0).
        verbose (bool): Enables detailed printing and visualization.

    Returns:
        list: A sorted list of UnitLoad objects (real and mock) representing the task queue.
    """
    if not uls:
        print("No unit loads to process.")
        return []

    # Create all tasks with their deadlines and time windows
    all_tasks = []
    
    for ul in uls:
        # Add the original retrieval task
        if ul.retrieval_end is not None:
            retrieval_task = UnitLoad(
                id=ul.id,
                retrieval_start=ul.retrieval_start,
                retrieval_end=ul.retrieval_end,
                arrival_start=ul.arrival_start,
                arrival_end=ul.arrival_end
            )
            retrieval_task.is_mock = False
            retrieval_task.task_type = "RETRIEVAL"
            retrieval_task.deadline = ul.retrieval_end
            retrieval_task.tw_start = ul.retrieval_start
            all_tasks.append(retrieval_task)
        
        # Add a "mock" storage task if the UL is not yet stored
        if not getattr(ul, 'stored', False) and ul.arrival_start is not None:
            # For storage tasks, we use retrieval_start/end fields to store arrival times
            # but need to ensure retrieval_start >= 1 to satisfy UnitLoad validation
            storage_retrieval_start = max(1, ul.arrival_start)
            storage_task = UnitLoad(
                id=f"{ul.id}_mock",
                retrieval_start=storage_retrieval_start,
                retrieval_end=ul.arrival_end,
                arrival_start=None,
                arrival_end=None
            )
            storage_task.is_mock = True
            storage_task.task_type = "STORAGE"
            storage_task.real_ul_id = ul.id
            storage_task.deadline = ul.arrival_end
            storage_task.tw_start = ul.arrival_start
            all_tasks.append(storage_task)

    # --- Group tasks with enclosed time windows ---
    task_groups = []
    # Sort tasks by start time before grouping to ensure consistent group formation
    all_tasks.sort(key=lambda task: task.tw_start)
    for task in all_tasks:
        merged = False
        for group in task_groups:
            # A task can join a group if its TW encloses or is enclosed by any member of the group
            can_merge = False
            for member in group:
                # Check for enclosure in either direction
                encloses = (task.tw_start <= member.tw_start and task.deadline >= member.deadline)
                is_enclosed = (member.tw_start <= task.tw_start and member.deadline >= task.deadline)
                if encloses or is_enclosed:
                    can_merge = True
                    break
            
            if can_merge:
                group.append(task)
                merged = True
                break
        if not merged:
            task_groups.append([task])

    # --- Sort groups by the earliest deadline within each group ---
    task_groups.sort(key=lambda group: min(task.deadline for task in group))

    # --- Flatten the groups and assign priorities ---
    sorted_tasks = []
    priority = 1
    group_priority_map = {}

    for i, group in enumerate(task_groups):
        # Sort tasks within the group by deadline (EDD)
        group.sort(key=lambda task: task.deadline)
        group_priority_map[i + 1] = [t.id for t in group]
        for task in group:
            task.set_priority(i + 1) # All tasks in a group get the same group priority
            sorted_tasks.append(task)
        
    if verbose:
        print("\n--- Enhanced EDD Task Queue Generation ---")
        print(f"Total tasks created: {len(sorted_tasks)}")
        print(f"Number of task groups: {len(task_groups)}")
        print("-" * 60)
        for i, group in enumerate(task_groups):
             min_deadline = min(t.deadline for t in group)
             print(f"Group {i+1} (Sort Key/Min Deadline: {min_deadline}):")
             for task in group:
                 print(f"  - Task {task.id:<12} | Type: {task.task_type:<9} | TW: [{task.tw_start}-{task.deadline}]")
        print("-" * 60)
        
        if draw_prio_image:
            draw_task_queue_timeline(sorted_tasks)

    return sorted_tasks


def tws_to_priorities(uls, verbose=False): 
    """
    Create a dictionary mapping unit load IDs to their priorities using Earliest Due Date (EDD) algorithm.
    EDD prioritizes tasks based on their deadlines - earliest deadline gets highest priority.
    This function is maintained for backward compatibility.
    """
    if not uls:
        print("No unit loads found to prioritize.")
        return {}

    # Create list of all tasks with their deadlines  
    all_tasks = []
    
    for ul in uls:
        # Add retrieval task
        if ul.retrieval_end is not None:
            all_tasks.append({
                'id': ul.id,
                'type': 'retrieval',
                'deadline': ul.retrieval_end,
                'ul': ul
            })
        
        # Add storage task (arrival deadline) if unit load needs to arrive
        if ul.arrival_start is not None and ul.arrival_start > 0:
            all_tasks.append({
                'id': f"{ul.id}_mock",
                'type': 'storage', 
                'deadline': ul.arrival_end,
                'ul': ul
            })
    
    # Sort by Earliest Due Date (EDD) - earliest deadline gets priority 1
    all_tasks.sort(key=lambda task: task['deadline'])
    
    # Create priority dictionary
    priorities = {}
    for i, task in enumerate(all_tasks):
        priorities[task['id']] = i + 1  # Priority 1 = highest priority (earliest deadline)
        task['ul'].set_priority(i + 1)
    
    if verbose:
        print("\n--- Earliest Due Date (EDD) Priority Assignment ---")
        print(f"Total tasks: {len(all_tasks)}")
        print("-" * 60)
        for i, task in enumerate(all_tasks):
            print(f"Priority {i+1:2d}: UL {task['id']:<12} | Type: {task['type']:<9} | Deadline: {task['deadline']:3d}")
        print("-" * 60)
        
        # Draw timeline with EDD priorities
        if len(uls) > 0:
            draw_edd_timeline(all_tasks, priorities)
    
    return priorities
