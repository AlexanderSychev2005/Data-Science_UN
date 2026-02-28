import collections
import time
import random
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model


def generate_jobshop_data(num_jobs, num_machines, seed=42):
    """
    Generates initial data (matrix of artifact batches)
    """
    random.seed(seed)
    jobs_data = []
    for _ in range(num_jobs):
        machines = list(range(num_machines))
        random.shuffle(
            machines
        )  # Different batches may have different stage sequencesв
        job = [(m, random.randint(1, 10)) for m in machines]
        jobs_data.append(job)
    return jobs_data


def solve_cp_sat(jobs_data, time_limit=10.0):
    """
    Classic Constraint Programming method (Google OR-Tools).
    """
    start_time = time.time()
    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    horizon = sum(task[1] for job in jobs_data for task in job)

    model = cp_model.CpModel()
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    # Initialize variables and intervals
    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine, duration = task[0], task[1]
            start_var = model.NewIntVar(0, horizon, f"start_{job_id}_{task_id}")
            end_var = model.NewIntVar(0, horizon, f"end_{job_id}_{task_id}")
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, f"interval_{job_id}_{task_id}"
            )
            all_tasks[job_id, task_id] = (start_var, end_var)
            machine_to_intervals[machine].append(interval_var)

    # Constraint: No overlapping tasks on the same machine
    for machine in range(machines_count):
        model.AddNoOverlap(machine_to_intervals[machine])

    # Constraint: Precedence inside a job batch
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(
                all_tasks[job_id, task_id + 1][0] >= all_tasks[job_id, task_id][1]
            )

    # Objective function: Minimize overall makespan
    obj_var = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        obj_var,
        [all_tasks[job_id, len(job) - 1][1] for job_id, job in enumerate(jobs_data)],
    )
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.Solve(model)
    return time.time() - start_time


def solve_mcda_heuristic(jobs_data):
    """
    Innovative Multicriteria Decision Analysis (MCDA) R&D heuristic.
    """
    start_time = time.time()
    num_jobs = len(jobs_data)
    num_machines = 1 + max(task[0] for job in jobs_data for task in job)

    # Track states for scoring
    job_progress = [0] * num_jobs
    machine_avail = [0] * num_machines
    job_avail = [0] * num_jobs
    job_remaining = [sum(t[1] for t in job) for job in jobs_data]

    schedule = collections.defaultdict(list)
    AssignedTask = collections.namedtuple("AssignedTask", "start job index duration")

    while True:
        available_tasks = []
        for job_id in range(num_jobs):
            task_id = job_progress[job_id]
            if task_id < len(jobs_data[job_id]):
                available_tasks.append((job_id, task_id, *jobs_data[job_id][task_id]))

        if not available_tasks:
            break

        max_duration = max(t[3] for t in available_tasks) if available_tasks else 1
        max_remaining = max(job_remaining) if job_remaining else 1

        best_score, best_task, best_start = float("inf"), None, 0

        # MCDA Scoring implementation
        for task in available_tasks:
            job_id, task_id, machine, duration = task
            earliest_start = max(job_avail[job_id], machine_avail[machine])

            # Criteria normalization
            f1 = duration / max_duration
            f2 = 1.0 - (job_remaining[job_id] / max_remaining)

            # Linear convolution (compromise scheme with weights)
            score = 0.3 * f1 + 0.4 * f2 + 0.3 * earliest_start

            if score < best_score:
                best_score, best_task, best_start = score, task, earliest_start

        # Assign the optimal task based on score
        job_id, task_id, machine, duration = best_task
        schedule[machine].append(AssignedTask(best_start, job_id, task_id, duration))

        # Update system status
        end_time = best_start + duration
        machine_avail[machine] = job_avail[job_id] = end_time
        job_progress[job_id] += 1
        job_remaining[job_id] -= duration

    return max(machine_avail), time.time() - start_time, schedule


def plot_gantt(schedule, makespan, title, filename):
    """
    Generates an adapted Gantt chart for archaeological pipeline.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20.colors

    # Restoration stage names
    stages = [
        "3D-Scanning",
        "Preprocessing",
        "ML-Restoration",
        "NLP-Analysis",
        "Expert-Historian",
    ]

    for machine, tasks in schedule.items():
        for task in tasks:
            ax.barh(
                machine,
                task.duration,
                left=task.start,
                color=colors[task.job % len(colors)],
                edgecolor="black",
            )
            ax.text(
                task.start + task.duration / 2,
                machine,
                f"B-{task.job}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=9,
            )

    ax.set_xlabel("Time (conditional units)")
    ax.set_ylabel("Artifact Processing Stages")
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels(stages[: len(schedule)])
    ax.set_title(f"{title} (Makespan: {makespan})")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    sizes = [(3, 3), (5, 5), (8, 5), (10, 5), (15, 5), (20, 5)]
    cp_times, mcda_times = [], []

    for n_jobs, n_machines in sizes:
        data = generate_jobshop_data(n_jobs, n_machines)

        t_cp = solve_cp_sat(data, time_limit=10.0)
        mk_mcda, t_mcda, sched = solve_mcda_heuristic(data)

        cp_times.append(t_cp)
        mcda_times.append(t_mcda)

        # Draw chart for 8 batches across 5 stages
        if n_jobs == 8 and n_machines == 5:
            plot_gantt(
                sched,
                mk_mcda,
                "Gantt Chart: Artifact Restoration Schedule (MCDA)",
                "gantt_history.png",
            )

    # Build complexity graph
    plt.figure(figsize=(9, 5))
    x_labels = [f"{j}x{m}" for j, m in sizes]
    plt.plot(x_labels, cp_times, marker="o", label="CP-SAT (Classic)", color="red")
    plt.plot(x_labels, mcda_times, marker="s", label="MCDA (R&D Model)", color="blue")
    plt.xlabel("Problem Size (Batches x Stages)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Computational Complexity Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.savefig("complexity_history.png")
    plt.close()
