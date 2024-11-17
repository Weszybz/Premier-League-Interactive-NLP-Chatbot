import requests
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Expanded dictionary with Premier League team aliases
team_aliases = {
    "Manchester United": ["Man U", "Man Utd", "United", "Manchester U"],
    "Manchester City": ["Man City", "City", "MCFC"],
    "Arsenal": ["Gunners", "Arsenal FC"],
    "Tottenham": ["Spurs", "Tottenham Hotspur"],
    "Chelsea": ["Blues", "Chelsea FC"],
    "Liverpool": ["Reds", "Liverpool FC"],
    "Newcastle": ["Magpies", "Toon", "Newcastle United"],
    "West Ham": ["Hammers", "West Ham United"],
    "Aston Villa": ["Villa", "AVFC"],
    "Wolves": ["Wolverhampton Wanderers"],
    "Brighton": ["Brighton & Hove Albion", "Seagulls"],
    "Leicester": ["Foxes", "Leicester City"],
    "Crystal Palace": ["Palace", "Eagles"],
    "Everton": ["Toffees", "Everton FC"],
    "Southampton": ["Saints"],
    "Nottingham Forest": ["Forest"],
    "Leeds": ["Leeds United", "Whites"],
    "Burnley": ["Clarets"],
    "Sheffield United": ["Blades", "Sheffield"],
    "Bournemouth": ["Cherries", "AFC Bournemouth"]
}

# Training data for intent classification
training_data = [
    ("Chelsea vs Arsenal", "current_season"),
    ("Man Utd vs Liverpool", "current_season"),
    ("Chelsea vs Arsenal in 2023", "past_season"),
    ("Liverpool vs Man City in 2021", "past_season"),
    ("Tottenham vs Chelsea", "current_season"),
    ("Arsenal vs Brighton in 2019", "past_season"),
]

texts, labels = zip(*training_data)


# Preprocess function
def preprocess_input(text):
    """Preprocess input text by converting to lowercase and removing punctuation."""
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)


# Pipeline for intent classification
intent_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Train the intent classifier
intent_pipeline.fit(texts, labels)


def map_alias_to_team_name(alias):
    """Map a team alias to its full team name."""
    for team, aliases in team_aliases.items():
        if alias in aliases:
            return team
    return alias


def search_event(event_name, season=None):
    """Fetch match data between two teams for a given season."""
    api_key = '771766'  # Replace with your actual API key
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/searchevents.php'

    def fetch_events(event_name, season):
        params = {'e': event_name, 's': season} if season else {'e': event_name}
        response_url = f"{base_url}?e={event_name}&s={season}"  # Debug URL
        print(f"Debug: Fetching events from URL: {response_url}")

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('event', []) or []
        else:
            print(f"Debug: Failed to fetch events, status code: {response.status_code}")
            return []

    # Properly formatted event names
    team1, team2 = event_name.split('_vs_')
    original_event_name = f"{team1}_vs_{team2}"
    flipped_event_name = f"{team2}_vs_{team1}"

    original_events = fetch_events(original_event_name, season)
    flipped_events = fetch_events(flipped_event_name, season)

    events = original_events + flipped_events
    sorted_events = sorted(events, key=lambda x: x.get('dateEvent', ''), reverse=True)
    return sorted_events[:2] if not season else sorted_events


def extract_match_info(user_input):
    """Extract teams and optionally a year from user input."""
    match_with_year = re.search(r'(.+?)\s+vs\s+(.+?)\s+in\s+(\d{4})', user_input, re.IGNORECASE)
    match_without_year = re.search(r'(.+?)\s+vs\s+(.+)', user_input, re.IGNORECASE)  # Updated regex

    if match_with_year:
        team1, team2, year = match_with_year.groups()
        season = f"{year}-{int(year) + 1}"
    elif match_without_year:
        team1, team2 = match_without_year.groups()
        season = None  # No specific season provided, implies current season
    else:
        return None, None, None

    # Map aliases and ensure proper formatting
    team1 = map_alias_to_team_name(team1.strip()).replace(' ', '_')
    team2 = map_alias_to_team_name(team2.strip()).replace(' ', '_')
    return team1, team2, season



def chatbot():
    """Main chatbot function to handle user queries."""
    print("ChatBot: Welcome to the Football Match Info Bot!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break

        user_input_cleaned = preprocess_input(user_input)
        intent = intent_pipeline.predict([user_input_cleaned])[0]

        team1, team2, season = extract_match_info(user_input)

        if team1 and team2:
            if intent == "past_season":
                print(f"Searching for past season matches: {season}")
            elif intent == "current_season":
                print("Searching for current season matches.")
                season = None  # Current season logic

            events = search_event(f"{team1}_vs_{team2}", season)
            print(f"intent: {intent}")
            print(f"{team1}_vs_{team2} in {season}")

            if events:
                for event in events:
                    print(f"\nEvent: {event.get('strHomeTeam', 'Unknown')} vs {event.get('strAwayTeam', 'Unknown')}")
                    print(f"League: {event.get('strLeague', 'Unknown')}")
                    print(f"Date: {event.get('dateEvent', 'Unknown')}")
                    print(f"Venue: {event.get('strVenue', 'Unknown')}")
                    print(f"Score: {event.get('intHomeScore', 'N/A')} - {event.get('intAwayScore', 'N/A')}")
            else:
                print(f"ChatBot: No matches found.")
        else:
            print("ChatBot: Please provide input in 'team1 vs team2' or 'team1 vs team2 in year' format.")


# Example usage
if __name__ == "__main__":
    chatbot()
