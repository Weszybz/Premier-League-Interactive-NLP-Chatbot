import requests
import re


def search_event(event_name, season=None):
    """Fetch match data between two teams for a given season."""
    api_key = '771766'  # Replace with your actual API key
    base_url = f'https://www.thesportsdb.com/api/v1/json/{api_key}/searchevents.php'

    def fetch_events(event_name, season):
        params = {'e': event_name, 's': season}
        response_url = f"{base_url}?e={event_name}&s={season}"  # Construct the URL for debugging
        print(f"Debug: Fetching events from URL: {response_url}")  # Debug log

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('event', [])  # Ensure it always returns a list
        else:
            print(f"Debug: Failed to fetch events, status code: {response.status_code}")
            return []

    # Search original and flipped event names
    original_events = fetch_events(event_name, season)
    flipped_events = fetch_events('_vs_'.join(event_name.split('_vs_')[::-1]), season)

    events = original_events + flipped_events  # Safe concatenation of lists

    return [
        {
            'home': event['strHomeTeam'],
            'away': event['strAwayTeam'],
            'league': event['strLeague'],
            'date': event['dateEvent'],
            'time': event['strTime'],
            'venue': event['strVenue'],
            'home_score': event.get('intHomeScore', 'N/A'),
            'away_score': event.get('intAwayScore', 'N/A'),
            'description': event.get('strDescriptionEN', 'No description available.')
        }
        for event in events
    ]


def extract_match_info(user_input):
    """Extracts teams and year from user input."""
    match = re.search(r'(.+?)\s+vs\s+(.+?)\s+in\s+(\d{4})', user_input, re.IGNORECASE)

    if match:
        team1, team2, year = match.groups()
        season = f"{year}-{int(year) + 1}"
        team1 = team1.strip().replace(' ', '_')
        team2 = team2.strip().replace(' ', '_')
        return team1, team2, season
    return None, None, None


def chatbot():
    """Main chatbot function to handle user queries."""
    print("ChatBot: Welcome to the Football Match Info Bot!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("ChatBot: Goodbye!")
            break

        team1, team2, season = extract_match_info(user_input)
        if team1 and team2 and season:
            events = search_event(f"{team1}_vs_{team2}", season)
            if events:
                for event in events:
                    print(f"\nEvent: {event['home']} vs {event['away']}")
                    print(f"League: {event['league']}")
                    print(f"Date: {event['date']}")
                    print(f"Time: {event['time']}")
                    print(f"Venue: {event['venue']}")
                    print(f"Score: {event['home_score']} - {event['away_score']}")
                    print(f"Description: {event['description']}")
            else:
                print(
                    f"ChatBot: No matches found for {team1.replace('_', ' ')} vs {team2.replace('_', ' ')} in {season}.")
        else:
            print("ChatBot: Please provide the query in the format 'team1 vs team2 in year'.")


# Example usage
if __name__ == "__main__":
    chatbot()
