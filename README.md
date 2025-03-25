# ⚽ Premier League Interactive NLP Chatbot

A conversational AI chatbot that helps football fans retrieve match results, check fixtures, and book match tickets using natural language queries. Built in Python with real-time data via TheSportsDB API.

---

## ⚽ Overview

This chatbot acts as a virtual football assistant, understanding user intent through Natural Language Processing and handling multi-turn conversations with contextual awareness.

### 🧠 Core Features

- **Intent Recognition**: Classifies user inputs like "book tickets", "match results", and "next fixtures"
- **Entity Extraction**: Identifies teams, dates, and match info using regex and name mapping
- **Small Talk Support**: Responds to greetings, casual queries, and conversational chit-chat
- **User Personalization**:
  - Remembers name and favourite team
  - Handles “our team” logic in follow-up questions
- **Ticket Booking Flow**:
  - Asks for teams, dates, seating, and quantity
  - Confirms bookings and adds to user record
- **Error Handling**:
  - Detects incomplete or invalid queries
  - Provides suggestions and prompt ideas

---

## 🎯 Purpose

> “Can a conversational agent effectively assist users with real-time football data and booking tasks using natural language?”

The project explores practical applications of NLP pipelines for sports-focused assistants with real-time API integration.

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.7+
- `scikit-learn`
- `requests`
- `dateutil`

### ▶ Run the Chatbot

```bash
python prem_bot.py
```

### 🗂 Project Structure
```bash
├── prem_bot.py              # Main chatbot loop with intent handling
├── data/                    # (Optional) folder for logs or saved models
├── README.md                # You are here!
```
### 🛠 Technologies Used
- `Python` for backend and logic
- `scikit-learn` for intent classification
- `TheSportsDB API` for real-time football data
- `Regex` and `dateutil` for parsing queries

### 🧠 Example Conversation
```text
Welcome to the Premier League Interactive NLP-based AI!
• Find match results for your favourite teams. Try 'Wolves vs Crystal Palace in 2021'
• Check upcoming fixtures. Try 'Chelsea vs Arsenal'
• Book tickets. Try 'I want to book tickets for Brighton vs Aston Villa'

You: My name is Aaron  
ChatBot: Nice to meet you, Aaron! What is your favourite team?  
You: City  
ChatBot: Got it. You are now a fan of Manchester City...

You: When is our next match?  
ChatBot: Aston Villa's next fixture is against Manchester City on 2024-12-21 at 12:30.

You: I want to book tickets for Brighton vs Palace on 5th April 2025  
ChatBot: Great! Would you like VIP or regular seating?
```
