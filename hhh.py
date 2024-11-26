import requests
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

user_database = {
    "Wesley": {"team": "Chelsea"}
}

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
    ("Scores of the Chelsea vs Arsenal game", "current_season"),
    ("Scores for the Chelsea vs Arsenal game", "current_season"),
    ("Show scores Arsenal vs Chelsea", "current_season"),
    ("Find recent Chelsea vs Arsenal", "current_season"),
    ("Results for Arsenal vs Chelsea in 2022", "past_season"),
    ("Game between Chelsea and Arsenal in 2023", "past_season"),
    ("Fixtures Chelsea vs Arsenal", "current_season"),
    ("When did Arsenal play Chelsea?", "current_season"),
    # Current season examples
    ("Chelsea vs Liverpool", "current_season"),
    ("Arsenal vs Tottenham", "current_season"),
    ("When will Chelsea play Liverpool?", "current_season"),
    ("Show Chelsea's current fixtures", "current_season"),

    # Past season examples
    ("Chelsea vs Liverpool in 2012", "past_season"),
    ("What were the results of Chelsea vs Liverpool in 2013?", "past_season"),
    ("Scores for Arsenal vs Tottenham in 2015", "past_season"),
    ("Show me past matches between Chelsea and Arsenal in 2020", "past_season"),

    ("What is my name?", "user_info"),
    ("What is my favorite team?", "user_info"),
    ("When is the next fixture?", "next_fixture"),
    ("Show me the next match", "next_fixture"),
    ("What is the next game?", "next_fixture"),
    ("My name is Wesley", "introduce_name"),
    ("Call me Alice", "introduce_name"),
    ("I am John", "introduce_name"),
    ("My name is Barry", "introduce_name"),
    ("Call me Alice", "introduce_name"),
    ("I am John", "introduce_name"),
    ("You can call me Sarah", "introduce_name"),
    ("I'm known as Peter", "introduce_name"),
    ("Hey, I'm Lily", "introduce_name"),
    ("People call me Mike", "introduce_name"),
    ("My name is Sarah", "introduce_name"),
    ("I go by Anna", "introduce_name"),
    ("Current season matches", "current_season"),
    ("Show me the fixtures for this season", "current_season"),
    ("What were the results in 2022?", "past_season"),
    ("Show results for last season", "past_season")
]

texts, labels = zip(*training_data)

# Preprocess function
def preprocess_input(text):
    """Preprocess input text by converting to lowercase and removing punctuation."""
    text = text.lower().strip()
    return re.sub(r'[^\w\s]', '', text)  # Removes all non-alphanumeric and non-space characters

# Pipeline for intent classification
intent_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Train the intent classifier
intent_pipeline.fit(texts, labels)

def map_alias_to_team_name(alias):
    """Map a team alias to its full team name using the team_aliases dictionary."""
    for team, aliases in team_aliases.items():
        if alias.lower() in map(str.lower, aliases):  # Match aliases case-insensitively
            return team
    return alias  # Return the alias as-is if no match is found

def is_valid_team(team_name):
    """Check if the team name or alias is in the team_aliases dictionary."""
    return any(team_name.lower() in map(str.lower, aliases) for aliases in team_aliases.values()) or team_name in team_aliases

def search_event(event_name, season=None, query_type="both"):
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

    # Filter events based on query type
    if query_type == "past":
        return sorted_events[1:2]  # Return the most recent past event
    elif query_type == "future":
        return sorted_events[:1]  # Return the next upcoming event
    else:  # Return both past and future events
        return sorted_events[:2] if not season else sorted_events

def extract_match_info(user_input):
    """Extract teams and optionally a year from user input, ignoring unnecessary phrases."""
    # Remove unnecessary phrases before extracting team names
    normalized_input = re.sub(
        r'\b(fixtures? with|scores?|results? for|game of|match between|game between|when did|when|will|the|game|show|find|recent)\b',
        '',
        user_input,
        flags=re.IGNORECASE
    ).strip()

    # Convert "and" to "vs" for compatibility
    normalized_input = normalized_input.replace(" and ", " vs ")
    normalized_input = normalized_input.replace(" play ", " vs ")

    # Match patterns with and without a year
    match_with_year = re.search(r'(.+?)\s+vs\s+(.+?)\s+in\s+(\d{4})', normalized_input, re.IGNORECASE)
    match_without_year = re.search(r'(.+?)\s+vs\s+(.+)', normalized_input, re.IGNORECASE)

    if match_with_year:
        team1, team2, year = match_with_year.groups()
        season = f"{year}-{int(year) + 1}"
    elif match_without_year:
        team1, team2 = match_without_year.groups()
        season = None  # Current season implied
    else:
        return None, None, None

    # Map aliases to full team names
    team1 = map_alias_to_team_name(team1.strip()).replace(' ', '_')
    team2 = map_alias_to_team_name(team2.strip()).replace(' ', '_')
    return team1, team2, season


def detect_name_statement(user_input):
    """Detect if the user is providing their name or asking about their stored name."""
    name_pattern = re.search(r"(?:my name is|call me)\s+([a-zA-Z]+)", user_input, re.IGNORECASE)
    if name_pattern:
        return name_pattern.group(1).strip()  # Return the detected name
    elif re.search(r"what is my name\??", user_input, re.IGNORECASE):
        return "retrieve_name"  # Indicate a request to retrieve the name
    return None


def get_team_id(team_name):
    """Fetch the team ID for a given team name."""
    # Resolve the alias to the official team name
    resolved_name = map_alias_to_team_name(team_name)
    api_key = '771766'  # Replace with your actual API key
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/searchteams.php'
    response = requests.get(base_url, params={"t": resolved_name})

    if response.status_code == 200:
        data = response.json()
        teams = data.get('teams', [])
        if teams:
            return teams[0].get('idTeam', None)  # Return the first match's idTeam
    return None

def get_next_fixture_by_id(team_id):
    """Fetch the next fixture for a team using the team ID."""
    api_key = '771766'  # Replace with your actual API key
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/eventsnext.php'
    response = requests.get(base_url, params={"id": team_id})

    if response.status_code == 200:
        data = response.json()
        events = data.get('events', [])
        if events:
            next_event = events[0]  # Get the first upcoming event
            return {
                "home": next_event.get('strHomeTeam', 'Unknown'),
                "away": next_event.get('strAwayTeam', 'Unknown'),
                "date": next_event.get('dateEvent', 'Unknown'),
                "venue": next_event.get('strVenue', 'Unknown'),
                "league": next_event.get('strLeague', 'Unknown'),
                "time": next_event.get('strTime', 'Unknown')
            }
    return None


# Chatbot function with intents
def chatbot():
    """Main chatbot function to handle user queries."""
    print("ChatBot: Welcome to the Football Match Info Bot!")
    user_name = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break

        # Predict the intent
        user_input_cleaned = preprocess_input(user_input)
        intent = intent_pipeline.predict([user_input_cleaned])[0]
        handled = False  # Tracks whether the intent was successfully handled

        # Determine query type based on user input
        query_type = "both"
        if "when did" in user_input_cleaned:
            query_type = "past"
        elif "when will" in user_input_cleaned:
            query_type = "future"

        # Handle introduce_name intent
        if intent == "introduce_name":
            name_match = re.search(r"my name is (\w+)|call me (\w+)|i am (\w+)|i go by (\w+)", user_input_cleaned,
                                   re.IGNORECASE)
            if name_match:
                user_name = name_match.group(1) or name_match.group(2) or name_match.group(3) or name_match.group(4)
                if user_name in user_database:
                    print(f"ChatBot: Welcome back, {user_name}!")
                else:
                    print(f"ChatBot: Nice to meet you, {user_name}!")
                    while True:
                        print("ChatBot: What is your favorite team?")
                        favorite_team = input("You: ").strip()
                        if is_valid_team(favorite_team):
                            resolved_team = map_alias_to_team_name(favorite_team)
                            user_database[user_name] = {"team": resolved_team}
                            print(f"ChatBot: Got it, {user_name}. You are now a fan of {resolved_team}.")
                            break
                        else:
                            print("ChatBot: Please pick a valid Premier League team from the following:")
                            print(", ".join(team_aliases.keys()))
                handled = True

        # Handle user_info intent
        if intent == "user_info" and not handled:
            if not user_name:
                print("ChatBot: I don't know your name yet. Please tell me!")
            else:
                print(
                    f"ChatBot: Your name is {user_name}, and your favorite team is {user_database[user_name]['team']}.")
            handled = True

        # Handle next_fixture intent
        if intent == "next_fixture" and not handled:
            if not user_name:
                print("ChatBot: What is your name?")
                user_name = input("You: ").strip()
                if user_name in user_database:
                    print(f"ChatBot: Welcome back, {user_name}!")
                else:
                    print(f"ChatBot: Nice to meet you, {user_name}!")
                    while True:
                        print("ChatBot: What is your favorite team?")
                        favorite_team = input("You: ").strip()
                        if is_valid_team(favorite_team):
                            resolved_team = map_alias_to_team_name(favorite_team)
                            user_database[user_name] = {"team": resolved_team}
                            print(f"ChatBot: Got it, {user_name}. You are now a fan of {resolved_team}.")
                            break
                        else:
                            print("ChatBot: Please pick a valid Premier League team from the following:")
                            print(", ".join(team_aliases.keys()))
            favorite_team = user_database[user_name]["team"]
            team_id = get_team_id(favorite_team)
            if team_id:
                next_fixture = get_next_fixture_by_id(team_id)
                if next_fixture:
                    print(f"\nChatBot: The next fixture for {favorite_team} is:")
                    print(f"{next_fixture['home']} vs {next_fixture['away']}")
                    print(f"League: {next_fixture['league']}")
                    print(f"Date: {next_fixture['date']} at {next_fixture['time']}")
                    print(f"Venue: {next_fixture['venue']}")
                else:
                    print(f"ChatBot: Sorry, I couldn't find any upcoming fixtures for {favorite_team}.")
            else:
                print(
                    f"ChatBot: Sorry, I couldn't find the team ID for {favorite_team}. Please check the team name.")
            handled = True

        # Handle match-related queries
        if intent in ["current_season", "past_season"] and not handled:
            team1, team2, season = extract_match_info(user_input_cleaned)
            if team1 and team2:
                team1 = map_alias_to_team_name(team1).replace(" ", "_")
                team2 = map_alias_to_team_name(team2).replace(" ", "_")
                event_name = f"{team1}_vs_{team2}"
                events = search_event(event_name, season, query_type)

                if events:
                    print(
                        f"\nChatBot: Here are the results for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')}:")
                    for event in events:
                        print(f"- Date: {event.get('dateEvent', 'Unknown')}")
                        print(
                            f"  {event.get('strHomeTeam', 'Unknown')} vs {event.get('strAwayTeam', 'Unknown')}")
                        print(f"  Score: {event.get('intHomeScore', 'N/A')} - {event.get('intAwayScore', 'N/A')}")
                        print(f"  Venue: {event.get('strVenue', 'Unknown')}")
                        print(f"  League: {event.get('strLeague', 'Unknown')}\n")
                else:
                    print(
                        f"ChatBot: No matches found for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')} in {season if season else 'current season'}.")
            else:
                print("ChatBot: Please provide input in 'team1 vs team2' or 'team1 vs team2 in year' format.")
            handled = True

        # Fallback for unhandled intents
        if not handled:
            print("ChatBot: I didn't understand that. Try asking about matches, fixtures, or your name.")

            # Debug: Print intent for verification
        print(f"Debug: User Input: '{user_input}' - Predicted Intent: '{intent}'")
# Example usage
if __name__ == "__main__":
    chatbot()
