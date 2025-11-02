# --- Step 1: Import Libraries ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

# --- Step 2: Load and Prepare Data ---
print("Starting match predictor...")

# Define the relative path to your data file from the root project folder
DATA_FILE_PATH = 'data/EPL_2020_2023.csv'

try:
    # We now read in all columns, including the team names
    df = pd.read_csv(DATA_FILE_PATH) 
    print(f"File '{DATA_FILE_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: File not found at '{DATA_FILE_PATH}'.")
    print("Please make sure you are running this script from the 'match_predictor/' root folder.")
    exit()
except KeyError:
    print("Error: The data file seems to be missing 'home_name' or 'away_name' columns. Cannot proceed.")
    exit()

# 1. Convert 'date' column to datetime objects
# Remove ordinal suffixes (st, nd, rd, th) from dates before parsing
import re
df['date'] = df['date'].str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True)
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)

# 2. Sort the entire DataFrame by date
df = df.sort_values(by='date')

# 3. Rename 'class' to 'Result' for clarity
df = df.rename(columns={'class': 'Result'})

# We create a map so we can translate IDs (like '1') to names (like 'Arsenal') later
try:
    home_teams_map = df[['Home Team', 'home_name']].drop_duplicates()
    away_teams_map = df[['Away Team', 'away_name']].drop_duplicates()
    
    home_teams_map.columns = ['id', 'name']
    away_teams_map.columns = ['id', 'name']
    
    full_team_map_df = pd.concat([home_teams_map, away_teams_map]).drop_duplicates().set_index('id')
    team_name_map = full_team_map_df['name'].to_dict()
    print("Team ID-to-Name map created successfully.")
except Exception as e:
    print(f"Error creating team name map. Are 'Home Team', 'home_name', 'Away Team', 'away_name' columns present? Error: {e}")
    # We can still proceed, but the output will be IDs
    team_name_map = {}

print("Data loaded, sorted by date, and prepared.")

# --- Step 3: Feature Engineering ---
print("Starting feature engineering... (This may take a moment)")
# (This block remains unchanged)
features_list = []
teams = set(df['Home Team']).union(set(df['Away Team']))

season_stats = {team: {
    'matches_played': 0,
    'total_shots_on_target': 0,
    'total_corners': 0,
    'total_possession': 0,
    'total_chances': 0
} for team in teams}

for index, match in df.iterrows():
    home_team = match['Home Team']
    away_team = match['Away Team']
    match_date = match['date']
    
    past_matches = df[df['date'] < match_date]
    home_past_5 = past_matches[(past_matches['Home Team'] == home_team) | (past_matches['Away Team'] == home_team)].tail(5)
    away_past_5 = past_matches[(past_matches['Home Team'] == away_team) | (past_matches['Away Team'] == away_team)].tail(5)
    
    def get_form_points(team_matches, team_name):
        points = 0
        for _, row in team_matches.iterrows():
            if row['Home Team'] == team_name:
                if row['Result'] == 'h': points += 3
                elif row['Result'] == 'd': points += 1
            elif row['Away Team'] == team_name:
                if row['Result'] == 'a': points += 3
                elif row['Result'] == 'd': points += 1
        return points

    home_form_points = get_form_points(home_past_5, home_team)
    away_form_points = get_form_points(away_past_5, away_team)
    
    home_stats = season_stats[home_team]
    away_stats = season_stats[away_team]
    
    avg_home_shots = (home_stats['total_shots_on_target'] / home_stats['matches_played']) if home_stats['matches_played'] > 0 else 0
    avg_home_corners = (home_stats['total_corners'] / home_stats['matches_played']) if home_stats['matches_played'] > 0 else 0
    avg_home_possession = (home_stats['total_possession'] / home_stats['matches_played']) if home_stats['matches_played'] > 0 else 0
    avg_home_chances = (home_stats['total_chances'] / home_stats['matches_played']) if home_stats['matches_played'] > 0 else 0
    
    avg_away_shots = (away_stats['total_shots_on_target'] / away_stats['matches_played']) if away_stats['matches_played'] > 0 else 0
    avg_away_corners = (away_stats['total_corners'] / away_stats['matches_played']) if away_stats['matches_played'] > 0 else 0
    avg_away_possession = (away_stats['total_possession'] / away_stats['matches_played']) if away_stats['matches_played'] > 0 else 0
    avg_away_chances = (away_stats['total_chances'] / away_stats['matches_played']) if away_stats['matches_played'] > 0 else 0

    features_list.append({
        'date': match_date,
        'Home Team': home_team,
        'Away Team': away_team,
        'Result': match['Result'],
        'home_form_points': home_form_points,
        'away_form_points': away_form_points,
        'avg_home_shots': avg_home_shots,
        'avg_away_shots': avg_away_shots,
        'avg_home_corners': avg_home_corners,
        'avg_away_corners': avg_away_corners,
        'avg_home_possession': avg_home_possession,
        'avg_away_possession': avg_away_possession,
        'avg_home_chances': avg_home_chances,
        'avg_away_chances': avg_away_chances,
    })
    
    season_stats[home_team]['matches_played'] += 1
    season_stats[home_team]['total_shots_on_target'] += match['home_on']
    season_stats[home_team]['total_corners'] += match['home_corners']
    season_stats[home_team]['total_possession'] += match['home_possessions']
    season_stats[home_team]['total_chances'] += match['home_chances']
    
    season_stats[away_team]['matches_played'] += 1
    season_stats[away_team]['total_shots_on_target'] += match['away_on']
    season_stats[away_team]['total_corners'] += match['away_corners']
    season_stats[away_team]['total_possession'] += match['away_possessions']
    season_stats[away_team]['total_chances'] += match['away_chances']

features_df = pd.DataFrame(features_list)
print("Feature engineering complete.")

# --- Step 4: Final Prep and Data Splitting ---
# (This block remains unchanged)
model_ready_df = features_df.dropna()

feature_columns = [
    'home_form_points', 'away_form_points',
    'avg_home_shots', 'avg_away_shots',
    'avg_home_corners', 'avg_away_corners',
    'avg_home_possession', 'avg_away_possession',
    'avg_home_chances', 'avg_away_chances'
]
X = model_ready_df[feature_columns]
y = model_ready_df['Result']

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

X_test_with_names = model_ready_df.iloc[split_index:]

print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

# --- Step 5: Train the Random Forest Model ---
# (This block remains unchanged)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training the Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# --- Step 6: Test and Evaluate the Model ---
# (This block remains unchanged)
print("\n--- Model Evaluation Results (v1) ---")
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Average Accuracy: {accuracy * 100:.2f}%")

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['a (Away Win)', 'd (Draw)', 'h (Home Win)']))


# --- UPDATED: Step 7: Per-Team Predictability Analysis ---
print("\n--- Team Predictability Analysis ---")

# Create a DataFrame with the test results
test_results_df = X_test_with_names.copy()
test_results_df['prediction'] = y_pred
test_results_df['is_correct'] = (test_results_df['Result'] == test_results_df['prediction'])

# Get a list of all unique team IDs in the test set
test_teams = set(test_results_df['Home Team']).union(set(test_results_df['Away Team']))

team_accuracies = []

# Loop through each team and calculate their prediction accuracy
for team_id in test_teams:
    team_matches = test_results_df[
        (test_results_df['Home Team'] == team_id) | 
        (test_results_df['Away Team'] == team_id)
    ]
    
    if not team_matches.empty:
        team_acc = team_matches['is_correct'].mean()
        
        # --- NEW: Use the map to get the team name ---
        team_name = team_name_map.get(team_id, f"Unknown Team (ID: {team_id})")
        
        team_accuracies.append({
            'team_name': team_name,
            'accuracy': team_acc,
            'matches_in_test_set': len(team_matches)
        })

# Convert to a DataFrame for easy sorting
team_acc_df = pd.DataFrame(team_accuracies)
team_acc_df = team_acc_df.sort_values(by='accuracy', ascending=False)

# --- NEW: Print with team_name ---
print("\n--- Most Predictable Teams (Model Accuracy) ---")
print(team_acc_df[['team_name', 'accuracy', 'matches_in_test_set']].head(5).to_string(index=False))

print("\n--- Least Predictable Teams (Model Accuracy) ---")
print(team_acc_df[['team_name', 'accuracy', 'matches_in_test_set']].tail(5).to_string(index=False))

print("\n--- Demo Finished ---")
