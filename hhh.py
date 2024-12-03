import requests
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from dateutil.parser import parse

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

dialogue_state = {
    "current_intent": None,
    "team1": None,
    "team2": None,
    "date": None,
    "seating_type": None,
    "num_tickets": None,
    "ticket_available": False,
    "pending_task": None,  # e.g., "ask_for_seating" or "confirm_booking"
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

    # User Info
    ("What is my name?", "user_info"),
    ("Can you tell me my name?", "user_info"),
    ("What team do I support?", "user_info"),
    ("Do you know my name?", "user_info"),
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
    ("Show results for last season", "past_season"),
    # Booking examples
    ("I want to book tickets for Chelsea vs Arsenal on 2024-12-15", "book_ticket"),
    ("Book tickets for Liverpool vs Manchester United", "book_ticket"),
    ("Reserve 2 tickets for Tottenham vs Man City on 2023-11-20", "book_ticket"),
    ("Can I book a VIP ticket for Arsenal vs Chelsea?", "book_ticket"),
    ("I need tickets for Brighton vs Spurs", "book_ticket"),
    ("Buy tickets for Chelsea game", "book_ticket"),
    ("I want to book a match ticket for Liverpool", "book_ticket")
]

texts, labels = zip(*training_data)

# Preprocess function
def preprocess_input(text):
    """Preprocess input text by converting to lowercase and removing unnecessary punctuation."""
    text = text.lower().strip()
    # Keep numbers and alphanumeric characters (useful for dates and ticket counts)
    return re.sub(r'[^\w\s\d]', '', text)  # Remove non-alphanumeric characters except spaces

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
    """Extract teams and optionally a year or date from user input."""
    # Normalize the input by removing booking-related phrases
    normalized_input = re.sub(
        r'\b(i want to book|book|tickets?|for|on|match|game|play|fixtures?|results?|when did|when will|the|show|find|recent|in)\b',
        '',
        user_input,
        flags=re.IGNORECASE
    ).strip()

    # Extract date in the format YYYY-MM-DD, YYYYMMDD, or year (e.g., 2017 or written dates)
    date_match = re.search(r'\b(\d{4}-\d{2}-\d{2}|\d{8}|\d{4}|(?:\d{1,2}(?:st|nd|rd|th)?\s\w+\s\d{4}))\b', normalized_input)
    extracted_date = None
    season = None

    if date_match:
        date_text = date_match.group(0)
        try:
            if len(date_text) == 4:  # Handle standalone year as a season
                season = f"{date_text}-{int(date_text) + 1}"
            else:
                # Parse natural language or standard dates
                parsed_date = parse(date_text, fuzzy=True)
                extracted_date = parsed_date.strftime('%Y-%m-%d')
            normalized_input = normalized_input.replace(date_text, '').strip()  # Remove date/season from input
        except ValueError:
            pass  # Ignore invalid date parsing

    # Convert "and" or "play" to "vs" for compatibility
    normalized_input = normalized_input.replace(" and ", " vs ")
    normalized_input = normalized_input.replace(" play ", " vs ")

    # Extract teams from the remaining input
    match_teams = re.search(r'(.+?)\s+vs\s+(.+)', normalized_input, re.IGNORECASE)
    team1, team2 = None, None

    if match_teams:
        team1, team2 = match_teams.groups()

    # Map aliases to full team names
    if team1:
        team1 = map_alias_to_team_name(team1.strip()).replace(' ', '_')
    if team2:
        team2 = map_alias_to_team_name(team2.strip()).replace(' ', '_')

    return team1, team2, extracted_date or season

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


def handle_turn(user_input, state):
    """Handle a single turn of the conversation."""
    # Preprocess input
    user_input_cleaned = preprocess_input(user_input)
    print(f"Debug: Current State Before Processing: {state}")
    print(f"Debug: Cleaned User Input: {user_input_cleaned}")

    # Step 1: Handle Pending Task if Active
    if state.get("pending_task"):
        task = state["pending_task"]
        print(f"Debug: Handling Pending Task: {task}")

        if task == "ask_for_date":
            try:
                parsed_date = parse(user_input_cleaned, fuzzy=True).strftime('%Y-%m-%d')
                state["date"] = parsed_date
                state["pending_task"] = "ask_for_seating"
                print(f"Debug: Date Parsed Successfully: {parsed_date}")
                return (
                    f"Tickets are available for {state['team1'].replace('_', ' ')} vs {state['team2'].replace('_', ' ')} "
                    f"on {parsed_date}. What seating type would you like (VIP or regular)?")
            except ValueError:
                print(f"Debug: Failed to Parse Date: {user_input_cleaned}")
                return "I couldn't understand the date. Please provide it in a format like 'December 15, 2024' or '2024-12-15'."

        elif task == "ask_for_seating":
            seating_type = extract_seating_type(user_input_cleaned)
            if seating_type:
                state["seating_type"] = seating_type
                state["pending_task"] = "ask_for_num_tickets"
                print(f"Debug: Seating Type Selected: {seating_type}")
                return f"How many {seating_type} tickets would you like?"
            return "Please specify a seating type (VIP or regular)."

        elif task == "ask_for_num_tickets":
            num_tickets = extract_num_tickets(user_input_cleaned)
            if num_tickets:
                state["num_tickets"] = num_tickets
                state["pending_task"] = "confirm_booking"
                print(f"Debug: Number of Tickets Selected: {num_tickets}")
                return (f"Just to confirm, you want {num_tickets} {state['seating_type']} tickets for "
                        f"{state['team1'].replace('_', ' ')} vs {state['team2'].replace('_', ' ')} "
                        f"on {state['date']}. Is that correct?")
            return "Please specify the number of tickets."

        elif task == "confirm_booking":
            if "yes" in user_input_cleaned or "confirm" in user_input_cleaned:
                state.clear()
                print("Debug: Booking Confirmed")
                return "Great! Your booking is confirmed. You will receive your tickets via email. Enjoy the match!"
            elif "no" in user_input_cleaned or "cancel" in user_input_cleaned:
                state.clear()
                print("Debug: Booking Canceled")
                return "Your booking has been canceled."
            return "Please confirm your booking by saying 'yes' or cancel by saying 'no'."

        print(f"Debug: Unrecognized Pending Task: {task}")
        return "Something went wrong. Can you start over?"

    # Step 2: Predict Intent if No Pending Task
    intent = intent_pipeline.predict([user_input_cleaned])[0]
    state["current_intent"] = intent
    print(f"Debug: Predicted Intent: {intent}")

    if intent == "book_ticket":
        # Extract match details
        team1, team2, date = extract_match_info(user_input_cleaned)
        state.update({"team1": team1, "team2": team2, "date": date})

        if not team1 or not team2:
            state["pending_task"] = "ask_for_teams"
            return "Which teams are you booking tickets for?"
        if not date:
            state["pending_task"] = "ask_for_date"
            return "Which date is the match on?"

        state["pending_task"] = "ask_for_seating"
        return (f"Tickets are available for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')} "
                f"on {date}. What seating type would you like (VIP or regular)?")

    # Step 3: Fallback for Unrecognized Input
    return "I didn't understand that. Can you rephrase?"


# Helper functions
def extract_seating_type(user_input):
    """Extract seating type from user input."""
    if "vip" in user_input.lower():
        return "VIP"
    elif "regular" in user_input.lower():
        return "regular"
    return None


def extract_num_tickets(user_input):
    """Extract number of tickets from user input."""
    match = re.search(r'\b(\d+)\b', user_input)  # Match any number in the input
    return int(match.group(1)) if match else None

def check_ticket_availability(team1, team2, date):
    """Simulate checking ticket availability."""
    # For simplicity, assume all tickets are available
    return True

# Chatbot function with intents
def chatbot():
    """Main chatbot function to handle user queries."""
    print("ChatBot: Welcome to the Football Match Info Bot!")
    user_name = None
    state = {}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break

        # If state is not empty, stay in the transaction flow
        if state:
            response = handle_turn(user_input, state)
            print(f"ChatBot: {response}")

            # Debugging: Show state after processing
            print(f"Debug: Current State: {state}")
            continue

        # Predict the intent
        user_input_cleaned = preprocess_input(user_input)
        intent = intent_pipeline.predict([user_input_cleaned])[0]
        print(f"Debug: Predicted Intent: {intent}")
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
                    print("ChatBot: I don't know your name yet. Please tell me by saying 'My name is [Your Name]'.")
                else:
                    favorite_team = user_database.get(user_name, {}).get("team", "unknown")
                    print(f"ChatBot: Your name is {user_name}, and your favorite team is {favorite_team}.")
                handled = True

                # Booking Intent
        if intent == "book_ticket":
            response = handle_turn(user_input, state)
            print(f"ChatBot: {response}")
            handled = True
            print(f"Debug: Current State: {state}")
            continue


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
            print(f"Debug: Extracted Team1: {team1}, Team2: {team2}, Season: {season}")
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
        print(f"Debug: Current State: {state}")
        print(f"Debug: Cleaned User Input: '{user_input_cleaned}'")
        print(f"Debug: User Input: '{user_input}' - Predicted Intent: '{intent}'")

# Example usage
if __name__ == "__main__":
    chatbot()
