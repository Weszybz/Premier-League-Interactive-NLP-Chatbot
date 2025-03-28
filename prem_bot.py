import requests
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from dateutil.parser import parse

# User database to store user information
user_database = {
    "wesley": {"team": "Chelsea"}
}

# Dictionary with Premier League team aliases for normalisation
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

# State dictionary to manage the current interaction
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
    ("Manchester United vs Aston Villa", "current_season"),
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

    # Past season examples
    ("Chelsea vs Liverpool in 2012", "past_season"),
    ("What were the results of Chelsea vs Liverpool in 2013?", "past_season"),
    ("Scores for Arsenal vs Tottenham in 2015", "past_season"),
    ("Show me past matches between Chelsea and Arsenal in 2020", "past_season"),

    # User Info
    ("My name is Wesley", "introduce_name"),
    ("What is my name?", "user_info"),
    ("Can you tell me my name?", "user_info"),
    ("What team do I support?", "user_info"),
    ("Do you know my name?", "user_info"),
    ("What is my favourite team?", "user_info"),

    ("When is the next fixture?", "next_fixture"),
    ("Show me the next match", "next_fixture"),
    ("What is the next game?", "next_fixture"),
    ("Show Chelsea's current fixtures", "next_fixture"),
    ("When does Brighton play next?", "next_fixture"),
    ("When is our next game?", "next_fixture"),
    ("When is our game?", "next_fixture"),
    ("When do we play next?", "next_fixture"),
    ("When is the next Liverpool match?", "next_fixture"),
    ("When does my team play?", "next_fixture"),

    ("Show me Chelsea match results.", "last_fixture"),
    ("What were Chelsea last match results?", "last_fixture"),
    ("Results for Liverpool last game", "last_fixture"),
    ("What was Arsenal previous match?", "last_fixture"),
    ("When was Arsenal last match?", "last_fixture"),
    ("When was villa last game?", "last_fixture"),
    ("When did Villa last play", "last_fixture"),
    ("When was our last game?", "last_fixture"),

    ("Show top scorers", "out_of_scope"),
    ("Show player stats.", "out_of_scope"),
    ("Who is the best player?", "out_of_scope"),
    ("What is the weather?", "out_of_scope"),
    ("Tell me a joke", "out_of_scope"),
    ("Show team logos", "out_of_scope"),

    ("Tell me about Chelsea matches", "ambiguous_query"),
    ("What can you tell me about Liverpool?", "ambiguous_query"),
    ("Arsenal info", "ambiguous_query"),
    ("Show Chelsea", "ambiguous_query"),
    ("Chelsea details", "ambiguous_query"),

    ("My name is Wesley", "introduce_name"),
    ("What is my name?", "user_info"),
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

# Preprocess function to clean user input
def preprocess_input(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Pipeline for intent classification
intent_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', LogisticRegression())
])

# Train the intent classifier
intent_pipeline.fit(texts, labels)

# Train-test split to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
intent_pipeline.fit(X_train, y_train)

# Evaluate the pipeline
y_pred = intent_pipeline.predict(X_test)

# Function to map team aliases to their official names
def map_alias_to_team_name(alias):

    for team, aliases in team_aliases.items():
        if alias.lower() in map(str.lower, aliases):  # Match aliases case-insensitively
            return team
    return alias  # Return the alias as-is if no match is found

# Function to validate team names
def is_valid_team(team_name):

    # Replace underscores with spaces and normalize
    team_name = team_name.strip().lower()
    resolved_name = team_name.replace('_', ' ')
    # Debugging: Log what is being checked
    # print(f"Debug: Validating team '{team_name}' and '{resolved_name}'")

    # Check against full team names (case-insensitive)
    if resolved_name in (team.lower().replace('_', ' ').strip() for team in team_aliases.keys()):
        # print("Debug: Matched with full team name.")
        return True

    # Check against aliases (case-insensitive)
    for team, aliases in team_aliases.items():
        if team_name in map(str.lower, aliases):
            # print(f"Debug: Matched with alias of '{team}'.")
            return True

    # print("Debug: No match found.")
    return False

# Fetch match data between two teams for a given season using TheSportsDB API.
def search_event(event_name, season=None, query_type="both"):
    api_key = '771766'
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/searchevents.php'

    # Function to fetch events from the API
    def fetch_events(event_name, season):
        params = {'e': event_name, 's': season} if season else {'e': event_name}
        response_url = f"{base_url}?e={event_name}&s={season}"  # Debug URL

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('event', []) or []
        else:
            print(f"ChatBot: Failed to fetch events, status code: {response.status_code}")
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

# Function to extract match details (teams and date/season)
def extract_match_info(user_input):
    normalized_input = re.sub(
        r'\b(i want to book|book|tickets?|for|on|match|game|play|when|what|fixtures?|results?|did|will|were|of|does|the|show|find|recent|in|last|previous|is|was|next|me)\b',
        '',
        user_input,
        flags=re.IGNORECASE
    ).strip()

    # Extract date in the format YYYY-MM-DD, YYYYMMDD, or year (e.g., 2017 or natural dates)
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

    # Extract team names
    team1, team2 = None, None
    match_teams = re.search(r'(.+?)\s+vs\s+(.+)', normalized_input, re.IGNORECASE)

    if match_teams:
        team1, team2 = match_teams.groups()
        team1 = map_alias_to_team_name(team1.strip()).replace(' ', '_')
        team2 = map_alias_to_team_name(team2.strip()).replace(' ', '_')
    else:
        # Handle single team input
        team_match = re.search(r'(.+)', normalized_input, re.IGNORECASE)
        if team_match:
            team1 = map_alias_to_team_name(team_match.group(1).strip()).replace(' ', '_')

    return team1, team2, extracted_date or season

# Detect if the user is providing their name or asking about their stored name.
def detect_name_statement(user_input):
    name_pattern = re.search(r"(?:my name is|call me)\s+([a-zA-Z]+)", user_input, re.IGNORECASE)
    if name_pattern:
        return name_pattern.group(1).strip()  # Return the detected name
    elif re.search(r"what is my name\??", user_input, re.IGNORECASE):
        return "retrieve_name"  # Indicate a request to retrieve the name
    return None

# Fetch the team ID for a given team name using TheSportsDB API.
def get_team_id(team_name):
    resolved_name = map_alias_to_team_name(team_name)
    api_key = '771766'
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/searchteams.php'
    response = requests.get(base_url, params={"t": resolved_name})

    if response.status_code == 200:
        data = response.json()
        teams = data.get('teams', [])
        if teams:
            return teams[0].get('idTeam', None)  # Return the first match's idTeam
    return None

# Fetch the next fixture for a team using the team ID.
def get_next_fixture_by_id(team_id):
    api_key = '771766'
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/eventsnext.php'
    response = requests.get(base_url, params={"id": team_id})

    if response.status_code == 200:
        data = response.json()
        events = data.get('events', [])

        if events:
            next_event = events[0]
            return {
                "home": next_event.get('strHomeTeam', 'Unknown'),
                "away": next_event.get('strAwayTeam', 'Unknown'),
                "date": next_event.get('dateEvent', 'Unknown'),
                "venue": next_event.get('strVenue', 'Unknown'),
                "league": next_event.get('strLeague', 'Unknown'),
                "time": next_event.get('strTime', 'Unknown')
            }
    return None

# Fetch the last fixture for a team using the team ID.
def get_last_fixture_by_id(team_id):
    api_key = '771766'
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/eventslast.php'
    response = requests.get(base_url, params={"id": team_id})

    if response.status_code == 200:
        data = response.json()
        events = data.get('results', [])
        if events:
            last_event = events [0]
            return {
                "home": last_event.get('strHomeTeam', 'Unknown'),
                "away": last_event.get('strAwayTeam', 'Unknown'),
                "date": last_event.get('dateEvent', 'Unknown'),
                "venue": last_event.get('strVenue', 'Unknown'),
                "league": last_event.get('strLeague', 'Unknown'),
                "time": last_event.get('strTime', 'Unknown'),
                "intHomeScore": last_event.get('intHomeScore', 'Unknown'),
                "intAwayScore": last_event.get('intAwayScore', 'Unknown'),
            }
    return None

# Handle a single turn of the conversation based on user input and state.
def handle_turn(user_input, state):
    # Preprocess input
    user_input_cleaned = preprocess_input(user_input)

    # Check for exit or cancel command
    if user_input_cleaned in ["exit", "cancel"]:
        state.clear()
        return "Transaction cancelled. Let me know if you need help with anything else!"

    # Handle pending tasks in the conversation flow
    if state.get("pending_task"):
        task = state["pending_task"]

        if task == "ask_for_teams":
            team1, team2, _ = extract_match_info(user_input_cleaned)

            if not team1:
                return "I couldn't identify a team. Please specify valid team names like 'Chelsea' or 'Arsenal'."

            # if not is_valid_team(team1):
            #     return(f"ChatBot: {team1} is not a valid Premier League team.\nChatBot: Please specify a valid Premier League team from the following: " + ", ".join(team_aliases.keys()))

            if team1 and not team2:
                # Fetch the next fixture for the specified team
                resolved_team = map_alias_to_team_name(team1)
                team_id = get_team_id(resolved_team)
                if team_id:
                    next_fixture = get_next_fixture_by_id(team_id)
                else:
                    return f"I couldn't find any upcoming matches for {team1.replace('_', ' ')}. Please try again later."

                # Update the state with fixture details
                state.update({
                    "team1": next_fixture['home'].replace(' ', '_'),
                    "team2": next_fixture['away'].replace(' ', '_'),
                    "date": next_fixture.get('date', 'Unknown'),
                    "venue": next_fixture.get('venue', 'Unknown'),
                    "pending_task": "confirm_next_match"
                })

                # Offer to book tickets for the next match
                return (
                    f"The next match for {team1.replace('_', ' ')} is:\n"
                    f"{next_fixture['home']} vs {next_fixture['away']} on {next_fixture['date']} at {next_fixture['venue']}.\n"
                    f"Would you like to book tickets for this match?"
                )


            if not is_valid_team(team1):
                return(f"ChatBot: {team1} is not a valid Premier League team.\nChatBot: Please specify a valid Premier League team from the following: " + ", ".join(team_aliases.keys()))
            elif not is_valid_team(team2):
                return(f"ChatBot: {team2} is not a valid Premier League team.\nChatBot: Please specify a valid Premier League team from the following: " + ", ".join(team_aliases.keys()))
            state.update({"team1": team1, "team2": team2})

            event_name = f"{team1}_vs_{team2}"
            next_match = search_event(event_name, query_type="future")

            if next_match:
                match = next_match[0]
                state.update({
                    "date": match.get('dateEvent', 'Unknown'),
                    "venue": match.get('strVenue', 'Unknown'),
                    "pending_task": "confirm_next_match"
                })
                return (f"{team1.replace('_', ' ')} and {team2.replace('_', ' ')} are next playing at "
                        f"{state['venue']} on {state['date']}. Would you like to book these tickets?")
            else:
                state["pending_task"] = "ask_for_date"
                return f"I couldn't find the next match for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')}. When is the match?"

            if team2:
                return f"Got it! You want to book tickets for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')}. When is the match?"
            else:
                return f"Got it! You want to book tickets for {team1.replace('_', ' ')}. When is the match?"

        elif task == "ask_for_date":
            try:
                parsed_date = parse(user_input_cleaned, fuzzy=True).strftime('%Y-%m-%d')
                state["date"] = parsed_date
                state["pending_task"] = "ask_for_seating"
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
                return f"How many {seating_type} tickets would you like?"
            return "Please specify a seating type (VIP or regular)."

        elif task == "ask_for_num_tickets":
            num_tickets = extract_num_tickets(user_input_cleaned)
            if num_tickets:
                state["num_tickets"] = num_tickets
                state["pending_task"] = "confirm_booking"
                return (f"Just to confirm, you want {num_tickets} {state['seating_type']} tickets for "
                        f"{state['team1'].replace('_', ' ')} vs {state['team2'].replace('_', ' ')} "
                        f"on {state['date']}. Is that correct?")
            return "Please specify the number of tickets as a number."

        elif task == "confirm_booking":
            if "yes" in user_input_cleaned or "confirm" in user_input_cleaned:
                state.clear()
                return "Great! Your booking is confirmed. You will receive your tickets via email. Enjoy the match!"
            elif "no" in user_input_cleaned or "cancel" in user_input_cleaned:
                state.clear()
                return "Your booking has been cancelled."
            return "Please confirm your booking by saying 'yes' or cancel by saying 'no'."

        elif task == "confirm_next_match":
            if "yes" in user_input_cleaned or "confirm" in user_input_cleaned:
                state["pending_task"] = "ask_for_seating"
                return (
                    f"What seating type would you like (VIP or regular)?")
            elif "no" in user_input_cleaned in user_input_cleaned:
                state["pending_task"] = "ask_for_date"
                return "Which date is the match on?"
            else:
                return "Please confirm your booking by saying 'yes' or cancel by saying 'no'."
        return "Something went wrong. Can you start over?"

    # Predict Intent if No Pending Task
    intent = intent_pipeline.predict([user_input_cleaned])[0]
    state["current_intent"] = intent

    if intent == "book_ticket":
        # Extract match details
        team1, team2, _ = extract_match_info(user_input_cleaned)

        if not team1:
            state["pending_task"] = "ask_for_teams"
            return "Great! Which teams are you booking tickets for"

        if not is_valid_team(team1):
            print(f"ChatBot: {team1} is not a valid Premier League team.")
            print(f"ChatBot: Please specify a valid Premier League team from the following: " + ", ".join(team_aliases.keys()))

        if team1 and not team2:
            # Fetch the next fixture for the specified team
            resolved_team = map_alias_to_team_name(team1)
            team_id = get_team_id(resolved_team)
            if team_id:
                next_fixture = get_next_fixture_by_id(team_id)
            else:
                return f"I couldn't identify any teams. Please provide valid team names like 'Chelsea' or 'Chelsea vs Arsenal'."

            # Update the state with fixture details
            state.update({
                "team1": next_fixture['home'].replace(' ', '_'),
                "team2": next_fixture['away'].replace(' ', '_'),
                "date": next_fixture.get('date', 'Unknown'),
                "venue": next_fixture.get('venue', 'Unknown'),
                "pending_task": "confirm_next_match"
            })

            # Offer to book tickets for the next match
            return (
                f"The next match for {team1.replace('_', ' ')} is:\n"
                f"{next_fixture['home']} vs {next_fixture['away']} on {next_fixture['date']} at {next_fixture['venue']}.\n"
                f"Would you like to book tickets for this match?"
            )

        # Fetch the next match for two teams

        if not is_valid_team(team1):
            return (f"ChatBot: {team1} is not a valid Premier League team.\nChatBot: Please specify a valid Premier League team from the following: " + ", ".join(
                    team_aliases.keys()))
        elif not is_valid_team(team2):
            return (f"ChatBot: {team2} is not a valid Premier League team.\nChatBot: Please specify a valid Premier League team from the following: " + ", ".join(
                    team_aliases.keys()))
        state.update({"team1": team1, "team2": team2})
        event_name = f"{team1}_vs_{team2}"
        next_match = search_event(event_name, query_type="future")

        if next_match:
            match = next_match[0]
            state.update({
                "date": match.get('dateEvent', 'Unknown'),
                "venue": match.get('strVenue', 'Unknown'),
                "pending_task": "confirm_next_match"
            })
            return (f"{team1.replace('_', ' ')} and {team2.replace('_', ' ')} are next playing at "
                    f"{state['venue']} on {state['date']}. Would you like to book these tickets?")
        else:
            state["pending_task"] = "ask_for_date"
            return f"I couldn't find the next match for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')}. When is the match?"

    # Fallback for Unrecognized Input
    return "I didn't understand that. Can you rephrase?"

# Extract seating type (VIP or regular) from user input.
def extract_seating_type(user_input):
    if "vip" in user_input.lower():
        return "VIP"
    elif "regular" in user_input.lower():
        return "regular"
    return None

# Extract the number of tickets requested from user input.
def extract_num_tickets(user_input):
    match = re.search(r'\b(\d+)\b', user_input)  # Match any number in the input
    return int(match.group(1)) if match else None

#Handle Small Talk
def small_talk(user_input):
    user_input = user_input.lower().strip()

    # Small talk responses
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    farewells = ["bye", "goodbye", "see you", "take care"]
    how_are_you = ["how are you", "how are you doing"]
    thanks = ["thank you", "thanks", "appreciate it"]
    weather = ["how's the weather", "what's the weather like"]

    # Match user input to categories
    if any(greet in user_input for greet in greetings):
        return "Hello! How can I assist you today?"
    elif any(farewell in user_input for farewell in farewells):
        return "Goodbye! Have a great day!"
    elif any(how in user_input for how in how_are_you):
        return "I'm just a chatbot, but I'm here to help! How can I assist you?"
    elif any(ty in user_input for ty in thanks):
        return "You're welcome! Let me know if there's anything else I can help with."
    elif any(w in user_input for w in weather):
        return "I can't check the weather right now, but it's always a good day to talk about football!"

    # Default response if no match
    return "I didn't quite catch that. Can you rephrase or let me know how I can assist?"

# Chatbot function to handle user interactions
def chatbot():
    print(f"Welcome to the Premier League Interactive NLP-based AI! I can help you with the following:")
    print(f"•	Find match results for your favourite teams. Try 'Wolves vs Crystal Palace in 2021'")
    print(f"•	Check upcoming fixtures. Try 'Chelsea vs Arsenal'")
    print(f"•	Book tickets for matches. Try 'I want to book tickets for Brighton vs Aston Villa'")
    print(f"You can start by introducing yourself or asking about a match. How can I assist you today?")

    user_name = None
    state = {}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break

        # Handle small talk
        response = small_talk(user_input)
        if response != "I didn't quite catch that. Can you rephrase or let me know how I can assist?":
            print(f"ChatBot: {response}")
            continue

        # Process the user input
        if state:
            response = handle_turn(user_input, state)
            print(f"ChatBot: {response}")
            continue

        # Predict intent for new interactions
        user_input_cleaned = preprocess_input(user_input)
        if "what is my name" in user_input_cleaned:
            intent = "user_info"
        elif "my name is" in user_input_cleaned:
            intent = "introduce_name"
        # Check for "our" in the input and replace with the user's favorite team
        elif "our" in user_input_cleaned:
            if user_name and user_name in user_database:
                favourite_team = user_database[user_name].get("team", None)
                if favourite_team:
                    user_input = user_input.replace("our", favourite_team.lower())
                    user_input_cleaned = user_input_cleaned.replace("our", favourite_team.lower())

                    # Determine intent explicitly
                    if "next" in user_input_cleaned or "upcoming" in user_input_cleaned:
                        intent = "next_fixture"
                    elif "last" in user_input_cleaned or "previous" in user_input_cleaned:
                        intent = "last_fixture"
                    else:
                        # Default to ambiguous query for general "our" use cases
                        intent = "ambiguous_query"
            else:
                print("ChatBot: I don't know who 'our' refers to.")
                intent = "user_info"
        else:
            intent = intent_pipeline.predict([user_input_cleaned])[0]
        handled = False  # Tracks whether the intent was successfully handled

        # Determine query type based on user input
        query_type = "both"
        if "when did" in user_input_cleaned:
            query_type = "past"
        elif "when will" in user_input_cleaned:
            query_type = "future"

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
                        print("ChatBot: What is your favourite team?")
                        favourite_team = input("You: ").strip()
                        if is_valid_team(favourite_team):
                            resolved_team = map_alias_to_team_name(favourite_team)
                            user_database[user_name] = {"team": resolved_team}
                            print(f"ChatBot: Got it. You are now a fan of {resolved_team}. You can now ask:\n"
                                  f"    • 'When does {resolved_team} play next?'\n"
                                  f"    • 'Show me {resolved_team} match results.'\n"
                                  f"    • 'Book tickets for {resolved_team} next match.'")
                            break
                        else:
                            print(
                                "ChatBot: Please specify a valid Premier League team from the following: " + ", ".join(
                                    team_aliases.keys()))
                handled = True

        # Handle user_info intent
        if intent == "user_info" and not handled:
            if not user_name:
                print("ChatBot: I don't know your name yet. Please tell me by saying 'My name is [Your Name]'.")
            else:
                favourite_team = user_database.get(user_name, {}).get("team", "unknown")
                print(f"ChatBot: Your name is {user_name}, and your favourite team is {favourite_team}.")
            handled = True


        if intent == "book_ticket":
            response = handle_turn(user_input, state)
            print(f"ChatBot: {response}")
            handled = True
            continue



        if intent == "next_fixture" and not handled:
            # Extract the team name from user input
            team1, _, _ = extract_match_info(user_input_cleaned)

            if not team1:
                return "ChatBot: I couldn't identify a team. Please specify a valid team name like 'Arsenal' or 'Chelsea'."

            # Fetch the next fixture for the specified team
            team_id = get_team_id(team1)
            if team_id:
                next_fixture = get_next_fixture_by_id(team_id)
                if next_fixture:
                    print(f"ChatBot: {next_fixture['home']}'s next fixture is against {next_fixture['away']} in the {next_fixture['league']}. It's being played at {next_fixture['venue']} on {next_fixture['date']} at {next_fixture['time']}.")
                    print(f"ChatBot: By the way, I can also help you find {next_fixture['home']}’s last match. Type 'When was {next_fixture['home']} last game?'.”")

                else:
                    print(f"ChatBot: Sorry, I couldn't find any upcoming fixtures for {team1.replace('_', ' ')}.")
            else:
                print(
                    f"ChatBot: Sorry, I couldn't find the team ID for {team1.replace('_', ' ')}. Please check the team name.")
            handled = True

        if intent == "last_fixture" and not handled:
            # Extract the team name from user input
            team1, _, _ = extract_match_info(user_input_cleaned)

            if not is_valid_team(team1):
                print("ChatBot: Please specify a valid Premier League team from the following: " + ", ".join(team_aliases.keys()))


            if not team1:
                return "ChatBot: I couldn't identify a team. Please specify a valid team name like 'Arsenal' or 'Chelsea'."

            # Fetch the last fixture for the specified team
            team_id = get_team_id(team1)
            if team_id:
                last_fixture = get_last_fixture_by_id(team_id)
                if last_fixture:
                    print(f"ChatBot: {last_fixture['home']} last played {last_fixture['away']} on {last_fixture['date']} at {last_fixture['time']} and the score was {last_fixture['intHomeScore']} - {last_fixture['intAwayScore']}.")
                    print(f"ChatBot: By the way, I can also help you find {last_fixture['home']}’s upcoming game. Type 'When is {last_fixture['home']} next game?'.”")
                else:
                    print(f"ChatBot: Sorry, I couldn't find any upcoming fixtures for {team1.replace('_', ' ')}.")
            else:
                print(
                    f"ChatBot: Sorry, I couldn't find the team ID for {team1.replace('_', ' ')}. Please check the team name.")
            handled = True

        # Handle match-related queries
        if intent in ["current_season", "past_season"] and not handled:
            team1, team2, season = extract_match_info(user_input_cleaned)
            if team1 and team2:
                if not is_valid_team(team1):
                    print(
                        f"ChatBot: I didn’t catch the first team '{team1.replace('_', ' ')}'\nPlease specify a valid Premier League team from the following: " + ", ".join(
                            team_aliases.keys()))
                elif not is_valid_team(team2):
                    print(
                        f"ChatBot: I didn’t catch the second team '{team2.replace('_', ' ')}'\nPlease specify a valid Premier League team from the following: " + ", ".join(
                            team_aliases.keys()))
                else:
                    print(f"intent = {intent}")
                    team1 = map_alias_to_team_name(team1).replace(" ", "_")
                    team2 = map_alias_to_team_name(team2).replace(" ", "_")
                    event_name = f"{team1}_vs_{team2}"
                    events = search_event(event_name, season, query_type)

                    if events:
                        print(
                            f"\nChatBot: Here are the results for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')}:")
                        for event in events:
                            print(f"- Date: {event.get('dateEvent', 'Unknown')}")
                            print(f"  {event.get('strHomeTeam', 'Unknown')} vs {event.get('strAwayTeam', 'Unknown')}")
                            print(f"  Score: {event.get('intHomeScore', 'N/A')} - {event.get('intAwayScore', 'N/A')}")
                            print(f"  Venue: {event.get('strVenue', 'Unknown')}")
                            print(f"  League: {event.get('strLeague', 'Unknown')}\n")
                    else:
                        print(
                            f"ChatBot: No matches found for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')} in {season if season else 'current season'}.")
            else:
                print("ChatBot: I can assist with match results, fixtures, and ticket bookings. Try asking:")
                print("         •	‘Brighton vs Manchester United’")
                print("         •	‘Book tickets for Chelsea vs Wolves.’")
                print("         •	‘When does Liverpool play next?’")
            handled = True

        if intent == "ambiguous_query" and not handled:
            print(f"ChatBot: Do you want recent results, upcoming fixtures, or ticket information? Please specify so I can assist you better.")
            handled = True

        if intent == "out_of_scope" and not handled:
            print(f"ChatBot: I can’t provide player stats or unrelated information. Try asking about match results, fixtures, or ticket bookings. How can I assist you?" )
            handled = True

        # Fallback for unhandled intents
        if not handled:
            print("ChatBot: I can assist with match results, fixtures, and ticket bookings. Try asking:")
            print("         •	‘Brighton vs Manchester United’")
            print("         •	‘Book tickets for Chelsea vs Wolves.’")
            print("         •	‘When does Liverpool play next?’")

# Example usage
if __name__ == "__main__":
    chatbot()