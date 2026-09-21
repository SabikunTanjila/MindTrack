"""Dataset contract shared by training, validation, and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = 'Stress_Level'
LABELS = ['Low', 'Medium', 'High', 'Very High']
NUMERIC = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Study_Hours',
           'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
CATEGORICAL = ['Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use']
FEATURES = NUMERIC + CATEGORICAL
CLUSTER_FEATURES = NUMERIC[1:]
BOUNDS = {'Age': (18, 24), 'Avg_Daily_Usage_Hours': (0, 24),
          'Daily_Unlocks': (0, 2000), 'Study_Hours': (0, 24),
          'Physical_Activity_Hours': (0, 24), 'Sleep_Hours_Per_Night': (0, 24)}
DISPLAY_NAMES = {'Age': 'Age', 'Avg_Daily_Usage_Hours': 'Daily social-media hours',
                 'Daily_Unlocks': 'Daily phone unlocks', 'Study_Hours': 'Study hours',
                 'Physical_Activity_Hours': 'Physical-activity hours',
                 'Sleep_Hours_Per_Night': 'Sleep hours per night',
                 'Academic_Level': 'Academic level', 'Most_Used_Platform': 'Main platform',
                 'Purpose_Of_Use': 'Main purpose'}
EXCLUDED = ['Gender', 'Country', 'Mental_Health_Score']
