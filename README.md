# Blackjack Deluxe

## Overview
Blackjack Deluxe is a Python-based implementation of the classic casino card game, Blackjack. Built using the `pygame` library, this game offers a rich graphical interface, smooth animations, and a variety of features to enhance the player experience. The game includes a comprehensive achievement system, split and double-down options, insurance bets, and a visually appealing interface with animations for card dealing, chip betting, and particle effects.

## Features
- **Realistic Blackjack Gameplay**: Play against a dealer with standard Blackjack rules.
- **Achievement System**: Unlock over 30 unique achievements as you play, such as "First Win," "Blackjack!," and "Millionaire."
- **Split and Double Down**: Split your hand if you have two cards of the same value, or double your bet for a single additional card.
- **Insurance Bets**: Take insurance when the dealer shows an Ace.
- **Smooth Animations**: Enjoy card dealing, chip animations, and particle effects for a polished gaming experience.
- **Customizable Betting**: Place bets ranging from $10 to $1000 using chip values of $10, $50, $100, and $500.
- **Achievements Screen**: View all achievements, their descriptions, and your progress toward unlocking them.
- **Main Menu**: Start a new game, view achievements, or quit the game from the main menu.

## Requirements
- **Python 3.x**
- **Pygame Library**: Install using `pip install pygame`.

## How to Run
1. Ensure Python and Pygame are installed on your system.
2. Download or clone the repository containing the `blackjack.py` file.
3. Navigate to the directory containing the file.
4. Run the game using the command:
   ```bash
   python blackjack.py
   ```
5. Use the main menu to start a new game, view achievements, or quit.

## Controls
- **Mouse**: Click on buttons and chips to interact with the game.
- **Keyboard** (in the main menu):
  - **Up/Down Arrow Keys**: Navigate menu options.
  - **Enter**: Select an option.

## Gameplay Instructions
1. **Betting Phase**:
   - Click on chips to place your bet (minimum $10, maximum $1000).
   - Click "DEAL" to start the game once your bet is placed.
2. **Dealing Phase**:
   - The dealer deals two cards to you and two to themselves (one face down).
3. **Player Turn**:
   - **HIT**: Request another card.
   - **STAND**: End your turn.
   - **DOUBLE**: Double your bet and receive one more card (available on the first move).
   - **SPLIT**: Split your hand into two separate hands if you have two cards of the same value.
4. **Dealer Turn**:
   - The dealer reveals their face-down card and hits until their hand value is at least 17.
5. **Game Over**:
   - Compare your hand(s) to the dealer's hand to determine the winner.
   - Collect your winnings or start a new hand.

## Achievements
The game includes a variety of achievements to unlock, such as:
- **First Win**: Win your first hand.
- **Blackjack!**: Get a natural Blackjack (21 with your first two cards).
- **High Roller**: Place a bet of $500 or more.
- **Millionaire**: Reach $1,000,000 in total money.
- **Blackjack Legend**: Unlock all other achievements.

View your progress and unlocked achievements in the "Achievements" screen.

## Code Structure
- **Main Game Loop**: Handles game states (betting, dealing, player turn, dealer turn, game over).
- **Animation Classes**: Manage card, chip, text, and particle animations.
- **Achievement System**: Tracks player progress and unlocks achievements based on game events.
- **UI Elements**: Buttons, text rendering, and visual effects for an immersive experience.

## Assets
The game uses the following assets:
- **Card Images**: Located in the `assets/cards` folder.
- **Chip Images**: Located in the `assets/chips` folder.
- **Background Image**: A casino table image is used as the game background.

## Customization
- **Constants**: Modify `WIDTH`, `HEIGHT`, `CARD_WIDTH`, `CARD_HEIGHT`, and other constants in the code to adjust the game's appearance.
- **Achievements**: Add or modify achievements in the `ACHIEVEMENTS` dictionary.
- **Chip Values**: Adjust `CHIP_VALUES`, `MIN_BET`, and `MAX_BET` to change betting options.

## Known Issues
- Ensure all asset files are present in the correct directories to avoid placeholder images.
- The game assumes a standard deck of 52 cards. Custom decks or additional rules are not supported.

## Future Improvements
- Add sound effects and background music.
- Implement a save/load system for player progress and achievements.
- Introduce multiplayer or online leaderboards.
- Add more customization options for the table and card designs.

## Credits
- Developed using the `pygame` library.
- Card and chip images sourced from [OpenGameArt](https://opengameart.org/) (or other sources, if applicable).

## License
This project is open-source and available under the MIT License. Feel free to modify and distribute it as needed.

---

Enjoy playing **Blackjack Deluxe**! Good luck at the tables! 🃏🎰
