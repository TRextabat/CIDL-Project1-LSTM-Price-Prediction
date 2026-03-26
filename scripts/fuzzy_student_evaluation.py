#!/usr/bin/env python3
"""
Fuzzy Logic Application: Student Performance Evaluation System
===============================================================

This script implements a fuzzy inference system that evaluates student academic
performance based on three input variables (attendance rate, assignment score,
exam score) and produces a final grade as the output.

Course: Computational Intelligence and Deep Learning (CIDL)
         Ege University - Computer Engineering Department
         2025-2026 Spring Semester - Project 1, Section 1.iv

Dependencies:
    pip install scikit-fuzzy numpy matplotlib

Usage:
    python fuzzy_student_evaluation.py
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# ============================================================================
# 1. DEFINE FUZZY VARIABLES (ANTECEDENTS AND CONSEQUENT)
# ============================================================================

# Input variable 1: Attendance Rate (0-100%)
attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')

# Input variable 2: Assignment Score (0-100)
assignment = ctrl.Antecedent(np.arange(0, 101, 1), 'assignment')

# Input variable 3: Exam Score (0-100)
exam = ctrl.Antecedent(np.arange(0, 101, 1), 'exam')

# Output variable: Final Grade (0-100)
grade = ctrl.Consequent(np.arange(0, 101, 1), 'grade')

# ============================================================================
# 2. DEFINE MEMBERSHIP FUNCTIONS
# ============================================================================

# --- Attendance Rate Membership Functions (Triangular) ---
attendance['low'] = fuzz.trimf(attendance.universe, [0, 0, 40])
attendance['medium'] = fuzz.trimf(attendance.universe, [25, 50, 75])
attendance['high'] = fuzz.trimf(attendance.universe, [60, 100, 100])

# --- Assignment Score Membership Functions (Triangular) ---
assignment['low'] = fuzz.trimf(assignment.universe, [0, 0, 45])
assignment['medium'] = fuzz.trimf(assignment.universe, [30, 55, 80])
assignment['high'] = fuzz.trimf(assignment.universe, [65, 100, 100])

# --- Exam Score Membership Functions (Triangular) ---
exam['low'] = fuzz.trimf(exam.universe, [0, 0, 45])
exam['medium'] = fuzz.trimf(exam.universe, [30, 55, 80])
exam['high'] = fuzz.trimf(exam.universe, [65, 100, 100])

# --- Final Grade Membership Functions (Triangular, 5 levels) ---
grade['fail'] = fuzz.trimf(grade.universe, [0, 0, 35])
grade['poor'] = fuzz.trimf(grade.universe, [20, 35, 50])
grade['average'] = fuzz.trimf(grade.universe, [35, 50, 65])
grade['good'] = fuzz.trimf(grade.universe, [55, 70, 85])
grade['excellent'] = fuzz.trimf(grade.universe, [75, 100, 100])

# ============================================================================
# 3. DEFINE FUZZY RULES
# ============================================================================

# Rule 1: All high -> excellent
rule1 = ctrl.Rule(
    attendance['high'] & assignment['high'] & exam['high'],
    grade['excellent'],
    'All dimensions high -> Excellent'
)

# Rule 2: High attendance, high assignment, medium exam -> good
rule2 = ctrl.Rule(
    attendance['high'] & assignment['high'] & exam['medium'],
    grade['good'],
    'High att & assign, medium exam -> Good'
)

# Rule 3: High attendance, medium assignment, high exam -> good
rule3 = ctrl.Rule(
    attendance['high'] & assignment['medium'] & exam['high'],
    grade['good'],
    'High att & exam, medium assign -> Good'
)

# Rule 4: Medium attendance, high assignment, high exam -> good
rule4 = ctrl.Rule(
    attendance['medium'] & assignment['high'] & exam['high'],
    grade['good'],
    'Medium att, high assign & exam -> Good'
)

# Rule 5: All medium -> average
rule5 = ctrl.Rule(
    attendance['medium'] & assignment['medium'] & exam['medium'],
    grade['average'],
    'All dimensions medium -> Average'
)

# Rule 6: All low -> fail
rule6 = ctrl.Rule(
    attendance['low'] & assignment['low'] & exam['low'],
    grade['fail'],
    'All dimensions low -> Fail'
)

# Rule 7: Low attendance, medium assignment, medium exam -> poor
rule7 = ctrl.Rule(
    attendance['low'] & assignment['medium'] & exam['medium'],
    grade['poor'],
    'Low att, medium assign & exam -> Poor'
)

# Rule 8: High attendance, low assignment, low exam -> poor
rule8 = ctrl.Rule(
    attendance['high'] & assignment['low'] & exam['low'],
    grade['poor'],
    'High att but low assign & exam -> Poor'
)

# Rule 9: Low attendance, high assignment, high exam -> average
rule9 = ctrl.Rule(
    attendance['low'] & assignment['high'] & exam['high'],
    grade['average'],
    'Low att, high assign & exam -> Average'
)

# Rule 10: Medium attendance, low assignment, high exam -> average
rule10 = ctrl.Rule(
    attendance['medium'] & assignment['low'] & exam['high'],
    grade['average'],
    'Medium att, low assign, high exam -> Average'
)

# Rule 11: High attendance, medium assignment, medium exam -> good
rule11 = ctrl.Rule(
    attendance['high'] & assignment['medium'] & exam['medium'],
    grade['good'],
    'High att, medium assign & exam -> Good'
)

# ============================================================================
# 4. CREATE CONTROL SYSTEM AND SIMULATION
# ============================================================================

# Build the fuzzy control system with all rules
grade_control = ctrl.ControlSystem([
    rule1, rule2, rule3, rule4, rule5, rule6,
    rule7, rule8, rule9, rule10, rule11
])

# Create simulation instance
grade_simulation = ctrl.ControlSystemSimulation(grade_control)

# ============================================================================
# 5. COMPUTE RESULT FOR THE EXAMPLE INPUT
# ============================================================================

# Example student: attendance=75, assignment=82, exam=68
test_attendance = 75
test_assignment = 82
test_exam = 68

grade_simulation.input['attendance'] = test_attendance
grade_simulation.input['assignment'] = test_assignment
grade_simulation.input['exam'] = test_exam

# Perform fuzzy inference and defuzzification
grade_simulation.compute()

# Retrieve the defuzzified output
result = grade_simulation.output['grade']

# ============================================================================
# 6. PRINT RESULTS
# ============================================================================

print("=" * 70)
print("FUZZY STUDENT PERFORMANCE EVALUATION SYSTEM")
print("=" * 70)
print()
print("INPUT VALUES:")
print(f"  Attendance Rate : {test_attendance}%")
print(f"  Assignment Score: {test_assignment}/100")
print(f"  Exam Score      : {test_exam}/100")
print()
print("FUZZIFICATION OF INPUTS:")

# Show membership degrees for attendance
att_low = fuzz.interp_membership(attendance.universe,
                                  fuzz.trimf(attendance.universe, [0, 0, 40]),
                                  test_attendance)
att_med = fuzz.interp_membership(attendance.universe,
                                  fuzz.trimf(attendance.universe, [25, 50, 75]),
                                  test_attendance)
att_high = fuzz.interp_membership(attendance.universe,
                                   fuzz.trimf(attendance.universe, [60, 100, 100]),
                                   test_attendance)
print(f"  Attendance = {test_attendance}:")
print(f"    mu_low    = {att_low:.4f}")
print(f"    mu_medium = {att_med:.4f}")
print(f"    mu_high   = {att_high:.4f}")

# Show membership degrees for assignment
asgn_low = fuzz.interp_membership(assignment.universe,
                                   fuzz.trimf(assignment.universe, [0, 0, 45]),
                                   test_assignment)
asgn_med = fuzz.interp_membership(assignment.universe,
                                   fuzz.trimf(assignment.universe, [30, 55, 80]),
                                   test_assignment)
asgn_high = fuzz.interp_membership(assignment.universe,
                                    fuzz.trimf(assignment.universe, [65, 100, 100]),
                                    test_assignment)
print(f"  Assignment = {test_assignment}:")
print(f"    mu_low    = {asgn_low:.4f}")
print(f"    mu_medium = {asgn_med:.4f}")
print(f"    mu_high   = {asgn_high:.4f}")

# Show membership degrees for exam
exam_low = fuzz.interp_membership(exam.universe,
                                   fuzz.trimf(exam.universe, [0, 0, 45]),
                                   test_exam)
exam_med = fuzz.interp_membership(exam.universe,
                                   fuzz.trimf(exam.universe, [30, 55, 80]),
                                   test_exam)
exam_high = fuzz.interp_membership(exam.universe,
                                    fuzz.trimf(exam.universe, [65, 100, 100]),
                                    test_exam)
print(f"  Exam = {test_exam}:")
print(f"    mu_low    = {exam_low:.4f}")
print(f"    mu_medium = {exam_med:.4f}")
print(f"    mu_high   = {exam_high:.4f}")

print()
print("-" * 70)
print(f"DEFUZZIFIED OUTPUT (Centroid Method):")
print(f"  Final Grade = {result:.2f} / 100")
print("-" * 70)
print()

# Interpret the result
if result < 35:
    interpretation = "FAIL - The student has not met minimum requirements."
elif result < 50:
    interpretation = "POOR - The student shows significant deficiencies."
elif result < 65:
    interpretation = "AVERAGE - The student demonstrates acceptable performance."
elif result < 85:
    interpretation = "GOOD - The student shows solid academic achievement."
else:
    interpretation = "EXCELLENT - The student demonstrates outstanding performance."

print(f"INTERPRETATION: {interpretation}")
print()
print("ANALYSIS:")
print(f"  The student has strong attendance ({test_attendance}%) and a high")
print(f"  assignment score ({test_assignment}), with a moderate exam score")
print(f"  ({test_exam}). The fuzzy system integrates these factors through")
print(f"  its rule base and produces a final grade of {result:.2f}, which")
print(f"  falls in the 'good' performance category. The exam score, being")
print(f"  the weakest input, slightly pulls down the overall grade, while")
print(f"  excellent attendance and strong assignment work provide uplift.")
print()

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

# Determine output directory for figures
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
figures_dir = os.path.join(project_dir, 'outputs', 'figures')
os.makedirs(figures_dir, exist_ok=True)

# --- Figure 1: Input Membership Functions ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Attendance
ax = axes[0]
ax.plot(attendance.universe, fuzz.trimf(attendance.universe, [0, 0, 40]),
        'b-', linewidth=2, label='Low')
ax.plot(attendance.universe, fuzz.trimf(attendance.universe, [25, 50, 75]),
        'g-', linewidth=2, label='Medium')
ax.plot(attendance.universe, fuzz.trimf(attendance.universe, [60, 100, 100]),
        'r-', linewidth=2, label='High')
ax.axvline(x=test_attendance, color='k', linestyle='--', alpha=0.7,
           label=f'Input = {test_attendance}')
ax.set_title('Attendance Rate', fontsize=14, fontweight='bold')
ax.set_xlabel('Attendance (%)', fontsize=12)
ax.set_ylabel('Membership Degree', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Assignment
ax = axes[1]
ax.plot(assignment.universe, fuzz.trimf(assignment.universe, [0, 0, 45]),
        'b-', linewidth=2, label='Low')
ax.plot(assignment.universe, fuzz.trimf(assignment.universe, [30, 55, 80]),
        'g-', linewidth=2, label='Medium')
ax.plot(assignment.universe, fuzz.trimf(assignment.universe, [65, 100, 100]),
        'r-', linewidth=2, label='High')
ax.axvline(x=test_assignment, color='k', linestyle='--', alpha=0.7,
           label=f'Input = {test_assignment}')
ax.set_title('Assignment Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Membership Degree', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Exam
ax = axes[2]
ax.plot(exam.universe, fuzz.trimf(exam.universe, [0, 0, 45]),
        'b-', linewidth=2, label='Low')
ax.plot(exam.universe, fuzz.trimf(exam.universe, [30, 55, 80]),
        'g-', linewidth=2, label='Medium')
ax.plot(exam.universe, fuzz.trimf(exam.universe, [65, 100, 100]),
        'r-', linewidth=2, label='High')
ax.axvline(x=test_exam, color='k', linestyle='--', alpha=0.7,
           label=f'Input = {test_exam}')
ax.set_title('Exam Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Membership Degree', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

plt.suptitle('Input Membership Functions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path_1 = os.path.join(figures_dir, 'fuzzy_input_membership_functions.png')
plt.savefig(fig_path_1, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path_1}")

# --- Figure 2: Output Membership Functions ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(grade.universe, fuzz.trimf(grade.universe, [0, 0, 35]),
         'b-', linewidth=2, label='Fail')
ax2.plot(grade.universe, fuzz.trimf(grade.universe, [20, 35, 50]),
         'c-', linewidth=2, label='Poor')
ax2.plot(grade.universe, fuzz.trimf(grade.universe, [35, 50, 65]),
         'g-', linewidth=2, label='Average')
ax2.plot(grade.universe, fuzz.trimf(grade.universe, [55, 70, 85]),
         'orange', linewidth=2, label='Good')
ax2.plot(grade.universe, fuzz.trimf(grade.universe, [75, 100, 100]),
         'r-', linewidth=2, label='Excellent')
ax2.axvline(x=result, color='k', linestyle='--', linewidth=2, alpha=0.8,
            label=f'Defuzzified Output = {result:.2f}')
ax2.set_title('Output Membership Functions (Final Grade)', fontsize=14,
              fontweight='bold')
ax2.set_xlabel('Grade', fontsize=12)
ax2.set_ylabel('Membership Degree', fontsize=12)
ax2.legend(fontsize=10, loc='upper left')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path_2 = os.path.join(figures_dir, 'fuzzy_output_membership_functions.png')
plt.savefig(fig_path_2, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path_2}")

# --- Figure 3: Multiple Student Evaluation (Heatmap-style) ---
# Evaluate grades for varying attendance and exam scores (assignment fixed at 70)
att_range = np.arange(0, 101, 5)
exam_range = np.arange(0, 101, 5)
grade_surface = np.zeros((len(att_range), len(exam_range)))

for i, a in enumerate(att_range):
    for j, e in enumerate(exam_range):
        try:
            sim = ctrl.ControlSystemSimulation(grade_control)
            sim.input['attendance'] = a
            sim.input['assignment'] = 70  # Fixed assignment score
            sim.input['exam'] = e
            sim.compute()
            grade_surface[i, j] = sim.output['grade']
        except Exception:
            grade_surface[i, j] = np.nan

fig3, ax3 = plt.subplots(figsize=(10, 8))
im = ax3.imshow(grade_surface, origin='lower', aspect='auto',
                extent=[0, 100, 0, 100], cmap='RdYlGn', vmin=0, vmax=100)
cbar = plt.colorbar(im, ax=ax3, label='Final Grade')
ax3.set_xlabel('Exam Score', fontsize=12)
ax3.set_ylabel('Attendance Rate (%)', fontsize=12)
ax3.set_title('Final Grade Surface (Assignment Score = 70)',
              fontsize=14, fontweight='bold')

# Mark the example point
ax3.plot(test_exam, test_attendance, 'ko', markersize=10, markeredgewidth=2)
ax3.annotate(f'Example\n({test_exam}, {test_attendance})\nGrade={result:.1f}',
             xy=(test_exam, test_attendance),
             xytext=(test_exam - 25, test_attendance - 20),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
fig_path_3 = os.path.join(figures_dir, 'fuzzy_grade_surface.png')
plt.savefig(fig_path_3, dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path_3}")

print()
print("=" * 70)
print("All visualizations saved successfully.")
print("=" * 70)

# Show plots (comment out if running in non-interactive environment)
# plt.show()
