# Blackjack Deluxe

## Overview
Blackjack Deluxe is an advanced Python implementation of the classic casino card game, Blackjack. Built using the `pygame` library, this game offers a rich graphical interface with smooth animations, sound effects, and a variety of features to enhance the player experience. The game includes a comprehensive achievement system, split and double-down options, insurance bets, VIP rooms, and a visually appealing interface with animations for card dealing, chip betting, and particle effects.

## Features

### Core Gameplay
- **Realistic Blackjack Gameplay**: Play against a dealer with standard Blackjack rules
- **Multiple Game Modes**: Standard table and VIP room with higher stakes
- **Card Counting**: Track cards with a built-in card counter
- **Strategy Assistant**: Get real-time recommendations based on basic strategy and card counting

### VIP Experience
- **VIP Room**: High-stakes table with special rules (min $10,000 bet)
- **Exclusive VIP Table Design**: Premium visual experience
- **Special VIP Rules**: Double after split, surrender option

### Visual & Audio Enhancements
- **Smooth Animations**: Card dealing, chip animations, and particle effects
- **Sound Effects**: Realistic casino sounds for cards, chips, wins and losses
- **Themes**: Multiple visual themes for the strategy advisor
- **Dynamic Lighting**: Glowing buttons and interactive elements

### Betting System
- **Customizable Betting**: Place bets from $10 to $1,000,000,000,000
- **Chip Values**: $10, $50, $100, $250, $500, $1000
- **All-In Option**: Risk all your money on a single hand
- **Jackpot System**: Optional progressive jackpot (1% of each bet contributes)

### Advanced Features
- **Split Hands**: Split pairs into separate hands
- **Double Down**: Double your bet for one more card
- **Insurance**: Hedge against dealer blackjack
- **Achievement System**: 30+ unique achievements to unlock
- **Performance Analytics**: Track your playing statistics
- **Risk Assessment**: Advanced probabilistic modeling of game outcomes

### Strategy Tools
- **Multiple Strategy Modes**: Conservative, Balanced, and Aggressive
- **Probability Overlays**: See bust chances and expected values
- **Decision History**: Review your recent plays
- **Heatmaps**: Visualize your strategy performance

## Requirements
- **Python 3.x**
- **Pygame Library**: `pip install pygame`
- **Additional Packages**: numpy, scikit-learn, matplotlib (automatically installed if missing)

## How to Run
1. Ensure Python and required packages are installed
2. Download or clone the repository
3. Navigate to the directory containing the file
4. Run the game:
   ```bash
   python blackjack.py
