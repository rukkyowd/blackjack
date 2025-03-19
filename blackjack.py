import pygame
import random
import os
import time
import math
import logging
import sys
from pygame import gfxdraw

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
CARD_WIDTH, CARD_HEIGHT = 71, 96
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
GREEN = (34, 139, 34)
GOLD = (255, 215, 0)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
FONT = pygame.font.Font(None, 36)
CHIP_VALUES = [10, 50, 100, 500] 
MIN_BET, MAX_BET = 10, 500
INSURANCE_BUTTON = pygame.Rect(WIDTH//2 - 75, HEIGHT - 200, 150, 50)
SPLIT_BUTTON = pygame.Rect(WIDTH//2 + 200, HEIGHT - 100, 100, 50)

# Load images
def load_image(path, size=None):
    try:
        image = pygame.image.load(path)
        if size:
            image = pygame.transform.scale(image, size)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a placeholder image with text when loading fails
        surf = pygame.Surface(size if size else (CARD_WIDTH, CARD_HEIGHT))
        surf.fill((200, 200, 200))
        return surf

pygame.mixer.init()  # Initialize the mixer module

# Load assets

card_flip = pygame.mixer.Sound(os.path.join("assets", "sounds", "card_flip.wav"))
chip_place = pygame.mixer.Sound(os.path.join("assets", "sounds", "chip_place.wav"))
win = pygame.mixer.Sound(os.path.join("assets", "sounds", "win.wav"))
lose = pygame.mixer.Sound(os.path.join("assets", "sounds", "lose.wav"))
win.set_volume(0.5)
lose.set_volume(0.5)
CASINO_TABLE = load_image(os.path.join("assets", "casino_table.png"), (WIDTH, HEIGHT))
CARD_BACK = load_image(os.path.join("assets", "cards", "back.png"), (CARD_WIDTH, CARD_HEIGHT))

#Logging Console

log_file = "console_output.log"
open(log_file, "w").close()  # Manually clear the file

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),  # Overwrite file
        logging.StreamHandler(sys.stdout)  # Print to console
    ]
)

# Custom class to redirect print() and errors to logging
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            self.level(message)

    def flush(self):  # Needed for compatibility
        pass

# Redirect stdout and stderr
sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

# Example outputs
print("This is a normal print statement.")  # Gets logged
sys.stderr.write("This is an error message.\n")  # Gets logged

# Achievements
ACHIEVEMENTS = {
    "first_win": {
        "name": "First Win",
        "description": "Win your first hand of Blackjack.",
        "unlocked": False
    },
    "blackjack": {
        "name": "Blackjack!",
        "description": "Get a natural Blackjack (21 with your first two cards).",
        "unlocked": False
    },
    "high_roller": {
        "name": "High Roller",
        "description": "Place a bet of $500 or more.",
        "unlocked": False
    },
    "five_wins": {
        "name": "Five Wins",
        "description": "Win 5 hands of Blackjack.",
        "unlocked": False
    },
    "bust_artist": {
        "name": "Bust Artist",
        "description": "Bust 5 times in a row.",
        "unlocked": False
    },
    "millionaire": {
        "name": "Millionaire",
        "description": "Reach $1,000,000 in total money.",
        "unlocked": False
    },
    "lucky_streak": {
        "name": "Lucky Streak",
        "description": "Win 3 hands in a row.",
        "unlocked": False
    },
    "unlucky_streak": {
        "name": "Unlucky Streak",
        "description": "Lose 3 hands in a row.",
        "unlocked": False
    },
    "push_master": {
        "name": "Push Master",
        "description": "Tie with the dealer 5 times.",
        "unlocked": False
    },
    "risk_taker": {
        "name": "Risk Taker",
        "description": "Win a hand after hitting on 20.",
        "unlocked": False
    },
    "all_in": {
        "name": "All In",
        "description": "Bet all your money on a single hand.",
        "unlocked": False
    },
    "chip_collector": {
        "name": "Chip Collector",
        "description": "Collect $10,000 in total winnings.",
        "unlocked": False
    },
    "double_down_daredevil": {
        "name": "Double Down Daredevil",
        "description": "Double down and win 5 times.",
        "unlocked": False
    },
    "small_bettor": {
        "name": "Small Bettor",
        "description": "Place 10 bets of $10 or less.",
        "unlocked": False
    },
    "big_spender": {
        "name": "Big Spender",
        "description": "Place 10 bets of $500 or more.",
        "unlocked": False
    },
    "perfect_pair": {
        "name": "Perfect Pair",
        "description": "Be dealt two Aces.",
        "unlocked": False
    },
    "lucky_7s": {
        "name": "Lucky 7s",
        "description": "Be dealt three 7s in a single hand.",
        "unlocked": False
    },
    "royal_flush": {
        "name": "Royal Flush",
        "description": "Be dealt a King and Queen of the same suit.",
        "unlocked": False
    },
    "lowballer": {
        "name": "Lowballer",
        "description": "Win a hand with a total of 5 or less.",
        "unlocked": False
    },
    "ace_master": {
        "name": "Ace Master",
        "description": "Win 10 hands with an Ace in your hand.",
        "unlocked": False
    },
    "dealers_nightmare": {
        "name": "Dealer's Nightmare",
        "description": "Make the dealer bust 5 times.",
        "unlocked": False
    },
    "dealers_best_friend": {
        "name": "Dealer's Best Friend",
        "description": "Lose 5 hands to the dealer's Blackjack.",
        "unlocked": False
    },
    "dealers_bane": {
        "name": "Dealer's Bane",
        "description": "Win 5 hands where the dealer busts.",
        "unlocked": False
    },
    "dealers_equal": {
        "name": "Dealer's Equal",
        "description": "Tie with the dealer 10 times.",
        "unlocked": False
    },
    "comeback_king": {
        "name": "Comeback King",
        "description": "Win a hand after being down to less than $10.",
        "unlocked": False
    },
    "card_counter": {
        "name": "Card Counter",
        "description": "Win 10 hands in a row without busting.",
        "unlocked": False
    },
    "no_risk_no_reward": {
        "name": "No Risk, No Reward",
        "description": "Win a hand without hitting.",
        "unlocked": False
    },
    "lucky_number_21": {
        "name": "Lucky Number 21",
        "description": "Win 21 hands in total.",
        "unlocked": False
    },
    "bust_free": {
        "name": "Bust-Free",
        "description": "Play 10 hands without busting.",
        "unlocked": False
    },
    "blackjack_legend": {
        "name": "Blackjack Legend",
        "description": "Unlock all other achievements.",
        "unlocked": False
    }
}

# Progress trackers for achievements that need counting
PROGRESS_TRACKERS = {
    "five_wins": 0,
    "lucky_streak": 0,
    "unlucky_streak": 0,
    "bust_artist": 0,
    "push_master": 0,
    "double_down_daredevil": 0,
    "small_bettor": 0,
    "big_spender": 0,
    "ace_master": 0,
    "dealers_nightmare": 0,
    "dealers_best_friend": 0,
    "dealers_bane": 0,
    "dealers_equal": 0,
    "card_counter": 0,
    "lucky_number_21": 0,
    "bust_free": 0,
    "chip_collector": 0
}

# Requirements for each trackable achievement
PROGRESS_REQUIREMENTS = {
    "five_wins": 5,
    "lucky_streak": 3,
    "unlucky_streak": 3,
    "bust_artist": 5,
    "push_master": 5,
    "double_down_daredevil": 5,
    "small_bettor": 10,
    "big_spender": 10,
    "ace_master": 10,
    "dealers_nightmare": 5,
    "dealers_best_friend": 5,
    "dealers_bane": 5,
    "dealers_equal": 10,
    "card_counter": 10,
    "lucky_number_21": 21,
    "bust_free": 10,
    "chip_collector": 10000
}

# Animation classes (unchanged)
class CardAnimation:
    def __init__(self, card, start_pos, end_pos, duration=0.5, flip_duration=0.3):
        self.card = card
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000  # Convert to milliseconds
        self.flip_duration = flip_duration * 1000  # Duration of the flip animation
        self.complete = False
        self.flip_progress = 0  # 0 to 1, where 0.5 is fully sideways
        self.is_flipping = False
        self.current_pos = start_pos

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True
            self.current_pos = self.end_pos
            return

        # Calculate progress (0 to 1)
        progress = elapsed / self.duration

        # Linear interpolation for position
        self.current_pos = (
            self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress,
            self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        )

        # Handle flip animation
        if elapsed < self.flip_duration:
            self.is_flipping = True
            self.flip_progress = elapsed / self.flip_duration
        else:
            self.is_flipping = False

    def draw(self, screen, front_image, back_image):
        if self.complete:
            screen.blit(front_image, self.end_pos)
            return

        if self.is_flipping:
            # Draw the flip animation
            if self.flip_progress < 0.5:
                # First half: show back image, scale horizontally
                scale_x = 1 - (self.flip_progress * 2)
                scaled_image = pygame.transform.scale(back_image, (int(CARD_WIDTH * scale_x), CARD_HEIGHT))
                screen.blit(scaled_image, (self.current_pos[0] + (CARD_WIDTH - scaled_image.get_width()) // 2, self.current_pos[1]))
            else:
                # Second half: show front image, scale horizontally
                scale_x = (self.flip_progress - 0.5) * 2
                scaled_image = pygame.transform.scale(front_image, (int(CARD_WIDTH * scale_x), CARD_HEIGHT))
                screen.blit(scaled_image, (self.current_pos[0] + (CARD_WIDTH - scaled_image.get_width()) // 2, self.current_pos[1]))
        else:
            # Draw the card at the current position without flipping
            screen.blit(front_image, self.current_pos)

class ChipAnimation:
    def __init__(self, value, start_pos, end_pos, duration=0.5):
        self.value = value
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000
        self.complete = False
        self.current_pos = start_pos

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True
            self.current_pos = self.end_pos
            return

        # Bouncy easing function
        progress = elapsed / self.duration
        if progress < 0.5:
            progress = 4 * progress * progress
        else:
            progress = -1 + (4 * (progress - 0.5) * (2 - progress))

        # Calculate current position with a slight arc
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress

        # Add arc effect
        y -= math.sin(progress * math.pi) * 100

        self.current_pos = (x, y)


class TextEffect:
    def __init__(self, text, position, color, duration=2.0, size_start=36, size_end=66):
        self.text = text
        self.position = position
        self.color = color
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000
        self.complete = False
        self.size_start = size_start
        self.size_end = size_end
        self.current_size = size_start
        self.alpha = 255

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True
            return

        progress = elapsed / self.duration

        # Size animation
        if progress < 0.3:
            # Grow effect
            size_progress = progress / 0.3
            self.current_size = self.size_start + (self.size_end - self.size_start) * size_progress
            self.alpha = 255
        else:
            # Fade out
            fade_progress = (progress - 0.3) / 0.7
            self.alpha = 255 * (1 - fade_progress)

    def draw(self, screen):
        if self.complete:
            return

        font = pygame.font.Font(None, int(self.current_size))
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(self.alpha)
        text_rect = text_surf.get_rect(center=self.position)
        screen.blit(text_surf, text_rect)

class ParticleSystem:
    def __init__(self, position, color, count=30, duration=1.0):
        self.particles = []
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000
        self.complete = False

        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 200)
            size = random.randint(3, 8)
            lifetime = random.uniform(0.5, 1.0)

            self.particles.append({
                'pos': position,
                'velocity': (math.cos(angle) * speed, math.sin(angle) * speed),
                'size': size,
                'color': color,
                'start_time': pygame.time.get_ticks(),
                'lifetime': lifetime * 1000
            })

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True

        for particle in self.particles:
            particle_elapsed = current_time - particle['start_time']

            if particle_elapsed >= particle['lifetime']:
                continue

            progress = particle_elapsed / particle['lifetime']

            # Update position
            particle['pos'] = (
                particle['pos'][0] + particle['velocity'][0] * 0.016,  # 16ms frame time
                particle['pos'][1] + particle['velocity'][1] * 0.016
            )

            # Add gravity
            particle['velocity'] = (
                particle['velocity'][0],
                particle['velocity'][1] + 100 * 0.016  # Gravity
            )

    def draw(self, screen):
        if self.complete:
            return

        current_time = pygame.time.get_ticks()

        for particle in self.particles:
            particle_elapsed = current_time - particle['start_time']

            if particle_elapsed >= particle['lifetime']:
                continue

            progress = particle_elapsed / particle['lifetime']
            alpha = 255 * (1 - progress)

            pos = (int(particle['pos'][0]), int(particle['pos'][1]))
            radius = int(particle['size'] * (1 - progress * 0.5))

            # Create a surface for this particle with alpha
            particle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color_with_alpha = (*particle['color'], int(alpha))

            # Draw antialiased circle
            gfxdraw.filled_circle(particle_surf, radius, radius, radius, color_with_alpha)
            gfxdraw.aacircle(particle_surf, radius, radius, radius, color_with_alpha)

            screen.blit(particle_surf, (pos[0] - radius, pos[1] - radius))

# Load card images
card_images = {}
suits = ['hearts', 'diamonds', 'clubs', 'spades']
values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king', 'ace']

for suit in suits:
    for value in values:
        filename = f"{value}_of_{suit}.png"
        card_images[(value, suit)] = load_image(os.path.join("assets", "cards", filename), (CARD_WIDTH, CARD_HEIGHT))

# Load chip images
chip_images = {}
for value in CHIP_VALUES:
    chip_images[value] = load_image(os.path.join("assets", "chips", f"chip_{value}.png"), (60, 60))

# Card values
card_values = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'jack': 10, 'queen': 10, 'king': 10, 'ace': 11
}

# Shuffle deck
def shuffle_deck():
    deck = [(v, s) for s in suits for v in values]
    random.shuffle(deck)
    return deck

# Deal a card
def deal_card(deck):
    if not deck:  # Check if deck is empty
        new_deck = shuffle_deck()
        deck.extend(new_deck)
    return deck.pop()

# Calculate hand value
def calculate_hand(hand):
    value = sum(card_values[card[0]] for card in hand)
    aces = sum(1 for card in hand if card[0] == 'ace')
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

# Move draw_glowing_button outside of calculate_hand
def draw_glowing_button(screen, rect, text, text_color, button_color, glow_color, glow_size=10, pulse=True):
    # Calculate pulse
    current_time = pygame.time.get_ticks() / 1000.0
    pulse_factor = 0.5 + math.sin(current_time * 5) * 0.5 if pulse else 1.0

    # Draw outer glow (multiple layers with decreasing alpha)
    for i in range(glow_size, 0, -1):
        alpha = int(180 * (i / glow_size) * pulse_factor)
        expanded_rect = pygame.Rect(
            rect.left - i, rect.top - i,
            rect.width + i * 2, rect.height + i * 2
        )
        glow_surf = pygame.Surface((expanded_rect.width, expanded_rect.height), pygame.SRCALPHA)
        # Ensure glow_color is a 3-tuple (RGB) before adding alpha
        if len(glow_color) == 3:
            glow_color_with_alpha = (*glow_color, alpha)
        else:
            glow_color_with_alpha = glow_color  # Assume it's already RGBA
        pygame.draw.rect(glow_surf, glow_color_with_alpha, 
                         (i, i, rect.width, rect.height), 
                         border_radius=12)
        screen.blit(glow_surf, (expanded_rect.left, expanded_rect.top))

    # Draw button body with gradient
    for i in range(rect.height):
        progress = i / rect.height
        color = [
            button_color[0] * (1 - progress * 0.3),
            button_color[1] * (1 - progress * 0.3),
            button_color[2] * (1 - progress * 0.3)
        ]
        pygame.draw.line(screen, color, 
                         (rect.left, rect.top + i), 
                         (rect.left + rect.width - 1, rect.top + i))

    # Draw button text
    font = pygame.font.Font(None, 36)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

# Calculate hand value
def calculate_hand(hand):
    value = sum(card_values[card[0]] for card in hand)
    aces = sum(1 for card in hand if card[0] == 'ace')
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

# Draw chips for betting
def draw_chips(screen, current_bet):
    x, y = WIDTH // 2 - 250, HEIGHT - 150
    for value in CHIP_VALUES:
        # Draw glow effect around chips
        glow_surf = pygame.Surface((80, 80), pygame.SRCALPHA)
        for i in range(20, 0, -1):
            alpha = 10 + i * 3
            pygame.draw.circle(glow_surf, (255, 215, 0, alpha), (40, 40), 30 + i)
        screen.blit(glow_surf, (x - 10, y - 10))

        # Draw chip
        screen.blit(chip_images[value], (x, y))

        # Draw value text with shadow
        shadow_text = FONT.render(f"+{value}", True, BLACK)
        text = FONT.render(f"+{value}", True, WHITE)
        screen.blit(shadow_text, (x + 16, y + 66))
        screen.blit(text, (x + 15, y + 65))
        x += 120

# Create card deal animation
def create_deal_animation(card, deck_pos, target_pos, delay=0):
    return CardAnimation(card, deck_pos, target_pos, duration=0.3)

# Function to handle unlocking achievements
def unlock_achievement(achievement_key, achievements_unlocked, text_effects):
    if not ACHIEVEMENTS[achievement_key]["unlocked"]:
        ACHIEVEMENTS[achievement_key]["unlocked"] = True
        achievements_unlocked.append(achievement_key)

        # Create text effect for achievement notification
        text_effects.append(TextEffect(
            f"Achievement Unlocked: {ACHIEVEMENTS[achievement_key]['name']}!", 
            (WIDTH // 2, HEIGHT // 2 + 100), 
            GOLD,
            duration=3.0
        ))

        # Also create a particle effect for visual flair
        particle_effects = ParticleSystem((WIDTH // 2, HEIGHT // 2 + 100), GOLD, count=50, duration=2.0)
        return particle_effects
    return None

def get_card_value(card):
    rank = card[0]  # Extract the rank from the card tuple
    if rank in ['jack', 'queen', 'king']:
        return 10
    elif rank == 'ace':
        return 11
    elif rank == '10':
        return 10
    else:
        try:
            return int(rank)  # For ranks 2-9
        except ValueError:
            print(f"Warning: Unexpected card format: {card}, rank: {rank}")
            return 0  # Default value to prevent crashes

# Function to check achievements
def check_achievements(game_state, result, player_hand, dealer_hand, player_money, current_bet, 
                      achievements_unlocked, text_effects, particle_systems, stats):
    # Extract stats for easier access
    win_count = stats["win_count"]
    bust_count = stats["bust_count"]
    total_winnings = stats["total_winnings"]
    push_count = stats["push_count"]
    consecutive_wins = stats["consecutive_wins"]
    consecutive_losses = stats["consecutive_losses"]
    consecutive_busts = stats["consecutive_busts"]

    particles = None

    # Check for achievements based on game state and result
    if result == "YOU WIN!":
        stats["win_count"] += 1
        stats["consecutive_wins"] += 1
        stats["consecutive_losses"] = 0
        stats["consecutive_busts"] = 0
        stats["total_winnings"] += current_bet

        # First win achievement
        if not ACHIEVEMENTS["first_win"]["unlocked"]:
            particles = unlock_achievement("first_win", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Five wins achievement
        if not ACHIEVEMENTS["five_wins"]["unlocked"]:
            PROGRESS_TRACKERS["five_wins"] += 1
            if PROGRESS_TRACKERS["five_wins"] >= PROGRESS_REQUIREMENTS["five_wins"]:
                particles = unlock_achievement("five_wins", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

        # Lucky streak achievement
        if stats["consecutive_wins"] >= 3 and not ACHIEVEMENTS["lucky_streak"]["unlocked"]:
            particles = unlock_achievement("lucky_streak", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Blackjack achievement
        if calculate_hand(player_hand) == 21 and len(player_hand) == 2 and not ACHIEVEMENTS["blackjack"]["unlocked"]:
            particles = unlock_achievement("blackjack", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Lowballer achievement
        if calculate_hand(player_hand) <= 5 and not ACHIEVEMENTS["lowballer"]["unlocked"]:
            particles = unlock_achievement("lowballer", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Ace master achievement
        if 'ace' in [card[0] for card in player_hand]:
            if not ACHIEVEMENTS["ace_master"]["unlocked"]:
                PROGRESS_TRACKERS["ace_master"] += 1
                if PROGRESS_TRACKERS["ace_master"] >= PROGRESS_REQUIREMENTS["ace_master"]:
                    particles = unlock_achievement("ace_master", achievements_unlocked, text_effects)
                    if particles:
                        particle_systems.append(particles)

        # Lucky number 21 achievement
        if not ACHIEVEMENTS["lucky_number_21"]["unlocked"]:
            PROGRESS_TRACKERS["lucky_number_21"] += 1
            if PROGRESS_TRACKERS["lucky_number_21"] >= PROGRESS_REQUIREMENTS["lucky_number_21"]:
                particles = unlock_achievement("lucky_number_21", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

    elif result == "BUST!":
        stats["bust_count"] += 1
        stats["consecutive_busts"] += 1
        stats["consecutive_wins"] = 0
        stats["consecutive_losses"] += 1

        # Bust artist achievement
        if stats["consecutive_busts"] >= 5 and not ACHIEVEMENTS["bust_artist"]["unlocked"]:
            particles = unlock_achievement("bust_artist", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Unlucky streak achievement
        if stats["consecutive_losses"] >= 3 and not ACHIEVEMENTS["unlucky_streak"]["unlocked"]:
            particles = unlock_achievement("unlucky_streak", achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

    elif result == "DEALER BUSTS!":
        stats["dealer_busts"] += 1
        stats["win_count"] += 1
        stats["consecutive_wins"] += 1
        stats["consecutive_losses"] = 0
        stats["consecutive_busts"] = 0
        stats["total_winnings"] += current_bet

        # Dealer's nightmare achievement
        if not ACHIEVEMENTS["dealers_nightmare"]["unlocked"]:
            PROGRESS_TRACKERS["dealers_nightmare"] += 1
            if PROGRESS_TRACKERS["dealers_nightmare"] >= PROGRESS_REQUIREMENTS["dealers_nightmare"]:
                particles = unlock_achievement("dealers_nightmare", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

    elif result == "PUSH":
        stats["push_count"] += 1
        stats["consecutive_wins"] = 0
        stats["consecutive_losses"] = 0
        stats["consecutive_busts"] = 0

        # Push master achievement
        if not ACHIEVEMENTS["push_master"]["unlocked"]:
            PROGRESS_TRACKERS["push_master"] += 1
            if PROGRESS_TRACKERS["push_master"] >= PROGRESS_REQUIREMENTS["push_master"]:
                particles = unlock_achievement("push_master", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

    # Check for bet-related achievements
    if current_bet >= 500 and not ACHIEVEMENTS["high_roller"]["unlocked"]:
        particles = unlock_achievement("high_roller", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    if current_bet >= 500:
        if not ACHIEVEMENTS["big_spender"]["unlocked"]:
            PROGRESS_TRACKERS["big_spender"] += 1
            if PROGRESS_TRACKERS["big_spender"] >= PROGRESS_REQUIREMENTS["big_spender"]:
                particles = unlock_achievement("big_spender", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

    if current_bet <= 10:
        if not ACHIEVEMENTS["small_bettor"]["unlocked"]:
            PROGRESS_TRACKERS["small_bettor"] += 1
            if PROGRESS_TRACKERS["small_bettor"] >= PROGRESS_REQUIREMENTS["small_bettor"]:
                particles = unlock_achievement("small_bettor", achievements_unlocked, text_effects)
                if particles:
                    particle_systems.append(particles)

    # Check for special card combinations
    if len(player_hand) == 2 and all(card[0] == 'ace' for card in player_hand) and not ACHIEVEMENTS["perfect_pair"]["unlocked"]:
        particles = unlock_achievement("perfect_pair", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    if len(player_hand) == 3 and all(card[0] == '7' for card in player_hand) and not ACHIEVEMENTS["lucky_7s"]["unlocked"]:
        particles = unlock_achievement("lucky_7s", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    # Check for royal flush (King and Queen of same suit)
    king_suit = None
    queen_suit = None
    for card in player_hand:
        if card[0] == 'king':
            king_suit = card[1]
        elif card[0] == 'queen':
            queen_suit = card[1]

    if king_suit and queen_suit and king_suit == queen_suit and not ACHIEVEMENTS["royal_flush"]["unlocked"]:
        particles = unlock_achievement("royal_flush", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    # Money-related achievements
    if player_money >= 1000000 and not ACHIEVEMENTS["millionaire"]["unlocked"]:
        particles = unlock_achievement("millionaire", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    if stats["total_winnings"] >= 10000 and not ACHIEVEMENTS["chip_collector"]["unlocked"]:
        particles = unlock_achievement("chip_collector", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    if player_money <= 10 and result == "YOU WIN!" and not ACHIEVEMENTS["comeback_king"]["unlocked"]:
        particles = unlock_achievement("comeback_king", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

# All-in achievement
    if current_bet == player_money and not ACHIEVEMENTS["all_in"]["unlocked"]:
        particles = unlock_achievement("all_in", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    # Check for all achievements unlocked
    all_unlocked = True
    for key, achievement in ACHIEVEMENTS.items():
        if key != "blackjack_legend" and not achievement["unlocked"]:
            all_unlocked = False
            break

    if all_unlocked and not ACHIEVEMENTS["blackjack_legend"]["unlocked"]:
        particles = unlock_achievement("blackjack_legend", achievements_unlocked, text_effects)
        if particles:
            particle_systems.append(particles)

    return particle_systems

def show_achievements_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack Achievements")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(BLACK)

        # Draw background
        bg_image = pygame.Surface((WIDTH, HEIGHT))
        bg_image.fill((20, 20, 50))
        for i in range(50):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            radius = random.randint(1, 3)
            pygame.draw.circle(bg_image, (255, 255, 255, 100), (x, y), radius)
        screen.blit(bg_image, (0, 0))

        # Draw title
        title_font = pygame.font.Font(None, 72)
        title_text = title_font.render("BlackJack Achievements", True, GOLD)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, 50))

        # Draw achievements grid
        keys = list(ACHIEVEMENTS.keys())
        rows, cols = 6, 5
        cell_width, cell_height = WIDTH // cols, 120

        for i, key in enumerate(keys):
            row = i // cols
            col = i % cols

            x = col * cell_width + 20
            y = row * cell_height + 150

            # Draw achievement box
            box_rect = pygame.Rect(x, y, cell_width - 40, cell_height - 20)

            if ACHIEVEMENTS[key]["unlocked"]:
                # Gold background for unlocked achievements
                pygame.draw.rect(screen, (50, 40, 0), box_rect, border_radius=8)
                pygame.draw.rect(screen, GOLD, box_rect, 2, border_radius=8)
                color = WHITE
            else:
                # Dark background for locked achievements
                pygame.draw.rect(screen, (40, 40, 40), box_rect, border_radius=8)
                pygame.draw.rect(screen, (100, 100, 100), box_rect, 2, border_radius=8)
                color = (150, 150, 150)

            # Draw achievement name
            name_font = pygame.font.Font(None, 28)
            name_text = name_font.render(ACHIEVEMENTS[key]["name"], True, color)
            screen.blit(name_text, (x + (cell_width - 40 - name_text.get_width())//2, y + 15))

            # Draw achievement description (wrapped text)
            desc_font = pygame.font.Font(None, 24)
            desc_text = ACHIEVEMENTS[key]["description"]
            words = desc_text.split()
            lines = []
            current_line = []

            for word in words:
                test_line = " ".join(current_line + [word])
                text_width = desc_font.size(test_line)[0]

                if text_width < cell_width - 60:
                    current_line.append(word)
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]

            if current_line:
                lines.append(" ".join(current_line))

            for j, line in enumerate(lines):
                line_surf = desc_font.render(line, True, color)
                screen.blit(line_surf, (x + (cell_width - 40 - line_surf.get_width())//2, y + 50 + j * 25))

        # Draw back button
        back_button = pygame.Rect(WIDTH//2 - 100, HEIGHT - 100, 200, 50)
        draw_glowing_button(screen, back_button, "Back to Game", WHITE, BLUE, (100, 100, 255))

        # Handle back button
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if back_button.collidepoint(mouse_pos):
                running = False

        pygame.display.flip()
        clock.tick(60)

# Define the main_menu() function before it is called
def main_menu():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack Deluxe - Menu")
    clock = pygame.time.Clock()

    # Initialize menu state
    menu_option = 0  # 0 = Play, 1 = Achievements, 2 = Quit

    # Menu loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    menu_option = (menu_option - 1) % 3
                elif event.key == pygame.K_DOWN:
                    menu_option = (menu_option + 1) % 3
                elif event.key == pygame.K_RETURN:
                    if menu_option == 0:
                        return "PLAY"
                    elif menu_option == 1:
                        return "ACHIEVEMENTS"
                    else:
                        return "QUIT"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                # Check if clicked on menu options
                play_button = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 50, 200, 50)
                achievements_button = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 20, 200, 50)
                quit_button = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 + 90, 200, 50)

                if play_button.collidepoint(mouse_pos):
                    return "PLAY"
                elif achievements_button.collidepoint(mouse_pos):
                    return "ACHIEVEMENTS"
                elif quit_button.collidepoint(mouse_pos):
                    return "QUIT"

        # Draw menu
        screen.fill(BLACK)

        # Draw animated background
        current_time = pygame.time.get_ticks() / 1000.0
        for i in range(50):
            x = (WIDTH//2) + math.cos(current_time * 0.5 + i * 0.2) * (WIDTH//3)
            y = (HEIGHT//2) + math.sin(current_time * 0.5 + i * 0.2) * (HEIGHT//3)
            size = 2 + math.sin(current_time + i) * 2
            color = (
                int(127 + 127 * math.sin(current_time * 0.7 + i * 0.1)),
                int(127 + 127 * math.sin(current_time * 0.5 + i * 0.1)),
                int(127 + 127 * math.sin(current_time * 0.3 + i * 0.1))
            )
            pygame.draw.circle(screen, color, (int(x), int(y)), int(size))

        # Draw title
        title_font = pygame.font.Font(None, 92)
        title_text = title_font.render("BLACKJACK DELUXE", True, GOLD)
        screen.blit(title_text, (WIDTH//2 - title_text.get_width()//2, HEIGHT//4))

        # Draw menu options
        options = ["Play Game", "Achievements", "Quit"]
        for i, option in enumerate(options):
            color = GOLD if i == menu_option else WHITE
            button_rect = pygame.Rect(WIDTH//2 - 100, HEIGHT//2 - 50 + i * 70, 200, 50)

            if i == menu_option:
                draw_glowing_button(screen, button_rect, option, BLACK, color, (*color, 128), pulse=True)
            else:
                draw_glowing_button(screen, button_rect, option, BLACK, color, (*color, 64), pulse=False)

        # Update display
        pygame.display.flip()
        clock.tick(60)

# Main game function
def main():
    dealer_welcome = pygame.mixer.Sound("assets/sounds/dealer_welcome.wav")
    dealer_welcome.play()
    # Set up display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack Deluxe")
    clock = pygame.time.Clock()
    

    # Initialize game state
    deck = shuffle_deck()
    player_hand = []
    dealer_hand = []
    player_money = 1000
    current_bet = 0
    game_state = "BETTING"  # game_state options : "BETTING", "DEALING", "INSURANCE", "PLAYER_TURN", "DEALER_TURN", "GAME_OVER"
    player_hands = [[]]  # List of hands (for splitting)
    current_hand_index = 0  # Index of the hand player is currently playing
    insurance_bet = 0  # Insurance bet amount

    # Animation tracking
    card_animations = []
    chip_animations = []
    text_effects = []
    particle_systems = []

    # Achievement and stats tracking
    achievements_unlocked = []
    stats = {
        "win_count": 0,
        "bust_count": 0,
        "dealer_busts": 0,
        "total_winnings": 0,
        "push_count": 0,
        "consecutive_wins": 0,
        "consecutive_losses": 0,
        "consecutive_busts": 0,
        "hands_without_bust": 0,
        "double_downs": 0,
        "splits": 0,
        "insurance_wins": 0
    }

    # Button rectangles
    hit_button = pygame.Rect(WIDTH//2 - 150, HEIGHT - 100, 100, 50)
    stand_button = pygame.Rect(WIDTH//2, HEIGHT - 100, 100, 50)
    double_button = pygame.Rect(WIDTH//2 + 150, HEIGHT - 100, 100, 50)
    deal_button = pygame.Rect(WIDTH//2 + 100, HEIGHT - 100, 100, 50)
    split_button = pygame.Rect(WIDTH//2 + 250, HEIGHT - 100, 100, 50)
    deal_button = pygame.Rect(WIDTH//2 + 100, HEIGHT - 100, 100, 50)
    insurance_button = pygame.Rect(WIDTH//2 - 200, HEIGHT - 100, 150, 50)
    no_insurance_button = pygame.Rect(WIDTH//2 + 50, HEIGHT - 100, 150, 50)

    # Results
    result = ""
    show_dealer_cards = False
    split_results = []  # Store results for each split hand

    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Mouse click event
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                # Handle betting state
                if game_state == "BETTING":
                    # Check if player clicked on chips
                    if HEIGHT - 150 <= mouse_pos[1] <= HEIGHT - 90:
                        for i, value in enumerate(CHIP_VALUES):
                            x = WIDTH // 2 - 250 + i * 120
                            if x <= mouse_pos[0] <= x + 60:
                                # Make sure bet doesn't exceed max or player's money
                                if current_bet + value <= MAX_BET and current_bet + value <= player_money:
                                    current_bet += value
                                    # Create chip animation
                                    chip_animations.append(ChipAnimation(
                                        value,
                                        (x, HEIGHT - 150),
                                        (WIDTH // 2, HEIGHT - 200)
                                    ))
                                    chip_place.play()

                    # Check if player clicked on deal button
                    if deal_button.collidepoint(mouse_pos) and current_bet >= MIN_BET:
                        game_state = "DEALING"

                        dealer_hit_or_stand = pygame.mixer.Sound("assets/sounds/hit_stand_double.wav")
                        dealer_hit_or_stand.play()

                        # Reset player hands for new game
                        player_hands = [[]]
                        current_hand_index = 0
                        split_results = []

                        # Deal initial cards with animations
                        deck_pos = (WIDTH - CARD_WIDTH - 50, 50)

                        # Player first card
                        player_card = deal_card(deck)
                        player_hands[0].append(player_card)
                        card_animations.append(create_deal_animation(
                            player_card,
                            deck_pos,
                            (WIDTH//2 - CARD_WIDTH - 10, HEIGHT//2 + 50)
                        ))
                        card_flip.play()

                        # Dealer first card
                        dealer_card = deal_card(deck)
                        dealer_hand.append(dealer_card)
                        card_animations.append(create_deal_animation(
                            dealer_card,
                            deck_pos,
                            (WIDTH//2 - CARD_WIDTH - 10, HEIGHT//2 - 150)
                        ))
                        card_flip.play()

                        # Player second card
                        player_card = deal_card(deck)
                        player_hands[0].append(player_card)
                        card_animations.append(create_deal_animation(
                            player_card,
                            deck_pos,
                            (WIDTH//2 + 10, HEIGHT//2 + 50)
                        ))
                        card_flip.play()

                        # Dealer second card (face down)
                        dealer_card = deal_card(deck)
                        dealer_hand.append(dealer_card)
                        card_animations.append(create_deal_animation(
                            dealer_card,
                            deck_pos,
                            (WIDTH//2 + 10, HEIGHT//2 - 150)
                        ))
                        card_flip.play()

                # Handle insurance option
                elif game_state == "INSURANCE":
                    # Check if player wants insurance
                    if insurance_button.collidepoint(mouse_pos):
                        # Insurance costs half the original bet
                        insurance_bet = current_bet // 2
                        player_money -= insurance_bet

                        # Create chip animation for insurance bet
                        chip_animations.append(ChipAnimation(
                            insurance_bet,
                            (WIDTH // 2 - 150, HEIGHT - 150),
                            (WIDTH // 2 - 150, HEIGHT - 300)
                        ))
                        chip_place.play()

                        # Move to player turn
                        game_state = "PLAYER_TURN"

                    # Check if player declines insurance
                    elif no_insurance_button.collidepoint(mouse_pos):
                        insurance_bet = 0
                        game_state = "PLAYER_TURN"

                # Handle player turn
                elif game_state == "PLAYER_TURN":
                    # Get current hand
                    player_hand = player_hands[current_hand_index]

                    # Hit button
                    if hit_button.collidepoint(mouse_pos):
                        player_card = deal_card(deck)
                        player_hand.append(player_card)

                        # Calculate position based on number of cards and current hand index
                        offset = len(player_hand) - 3
                        y_offset = current_hand_index * 100  # Offset for split hands
                        card_animations.append(create_deal_animation(
                            player_card,
                            (WIDTH - CARD_WIDTH - 50, 50),
                            (WIDTH//2 + CARD_WIDTH + offset * 30, HEIGHT//2 + 50 + y_offset)
                        ))

                        # Check if player busts
                        if calculate_hand(player_hand) > 21:
                            if current_hand_index < len(player_hands) - 1:
                                player_bust = pygame.mixer.Sound("assets/sounds/player_bust.wav")
                                player_bust.play()
                                # If there are more split hands, move to the next one
                                split_results.append("BUST!")
                                current_hand_index += 1
                            else:
                                # If this is the last hand, move to dealer's turn or game over
                                if len(player_hands) > 1:
                                    # If we have split hands, record the result for this hand
                                    split_results.append("BUST!")
                                    game_state = "DEALER_TURN"
                                    show_dealer_cards = True
                                else:
                                    game_state = "GAME_OVER"
                                    result = "BUST!"
                                    player_bust = pygame.mixer.Sound("assets/sounds/player_bust.wav")
                                    player_bust.play()
                                    show_dealer_cards = True
                                    player_money = int(player_money-current_bet)
                                    text_effects.append(TextEffect(
                                        result,
                                        (WIDTH//2, HEIGHT//2),
                                        RED
                                    ))
                                    lose.play()
                                    particle_systems.extend(check_achievements(
                                        game_state, result, player_hand, dealer_hand,
                                        player_money, current_bet, achievements_unlocked,
                                        text_effects, particle_systems, stats
                                    ) or [])

                    # Stand button
                    elif stand_button.collidepoint(mouse_pos):
                        if current_hand_index < len(player_hands) - 1:
                            # If there are more split hands, move to the next one
                            split_results.append("STAND")
                            standing = pygame.mixer.Sound("assets/sounds/standing.wav")
                            standing.play()
                            current_hand_index += 1
                        else:
                            # If this is the last hand, move to dealer's turn
                            if len(player_hands) > 1:
                                # If we have split hands, record the result for this hand
                                split_results.append("STAND")
                            game_state = "DEALER_TURN"
                            show_dealer_cards = True

                    # Double button (only available on first action)
                    elif double_button.collidepoint(mouse_pos) and len(player_hand) == 2:
                        if player_money >= current_bet:
                            player_money -= current_bet

                            # Add animation for doubling chips
                            chip_animations.append(ChipAnimation(
                                current_bet,
                                (WIDTH // 2 - 100, HEIGHT - 150),
                                (WIDTH // 2, HEIGHT - 200)
                            ))

                            double = pygame.mixer.Sound("assets/sounds/double_down.wav")
                            double.play()

                            chip_place.play()

                            # Deal one card to player
                            player_card = deal_card(deck)
                            player_hand.append(player_card)

                            # Calculate y offset for split hands
                            y_offset = current_hand_index * 100
                            card_animations.append(create_deal_animation(
                                player_card,
                                (WIDTH - CARD_WIDTH - 50, 50),
                                (WIDTH//2 + CARD_WIDTH, HEIGHT//2 + 50 + y_offset)
                            ))

                            # Record double down for achievements
                            stats["double_downs"] += 1

                            if current_hand_index < len(player_hands) - 1:
                                # If there are more split hands, move to the next one
                                split_results.append("DOUBLE")
                                current_hand_index += 1
                            else:
                                # If this is the last hand, move to dealer's turn
                                if len(player_hands) > 1:
                                    # If we have split hands, record the result for this hand
                                    split_results.append("DOUBLE")
                                game_state = "DEALER_TURN"
                                show_dealer_cards = True

                    # Split button (only if first two cards are the same value)
                    elif split_button.collidepoint(mouse_pos) and len(player_hand) == 2:
                        # Check if cards are the same value and player has enough money
                        card1_value = get_card_value(player_hand[0])
                        card2_value = get_card_value(player_hand[1])

                        if card1_value == card2_value and player_money >= current_bet:
                            # Take additional bet for the split hand
                            player_money -= current_bet

                            # Create a new hand with the second card
                            new_hand = [player_hand.pop()]
                            player_hands.append(new_hand)

                            split = pygame.mixer.Sound("assets/sounds/split_hand.wav")
                            split.play()

                            # Add animation for splitting chips
                            chip_animations.append(ChipAnimation(
                                current_bet,
                                (WIDTH // 2 - 100, HEIGHT - 150),
                                (WIDTH // 2 + 100, HEIGHT - 200)
                            ))
                            chip_place.play()

                            # Deal a new card to the first hand
                            player_card = deal_card(deck)
                            player_hand.append(player_card)
                            card_animations.append(create_deal_animation(
                                player_card,
                                (WIDTH - CARD_WIDTH - 50, 50),
                                (WIDTH//2 + 10, HEIGHT//2 + 50)
                            ))

                            # Deal a new card to the second hand
                            second_card = deal_card(deck)
                            player_hands[1].append(second_card)
                            card_animations.append(create_deal_animation(
                                second_card,
                                (WIDTH - CARD_WIDTH - 50, 50),
                                (WIDTH//2 + 10, HEIGHT//2 + 150)
                            ))

                            # Record split for achievements
                            stats["splits"] += 1

                # Handle game over state - start new game
                elif game_state == "GAME_OVER":
                    if hit_button.collidepoint(mouse_pos) or stand_button.collidepoint(mouse_pos):
                        # Reset for new hand
                        player_hands = [[]]
                        dealer_hand = []
                        current_hand_index = 0
                        current_bet = 0
                        insurance_bet = 0
                        game_state = "BETTING"
                        result = ""
                        split_results = []
                        show_dealer_cards = False

        # Update animations
        for animation in card_animations[:]:
            animation.update()
            if animation.complete:
                card_animations.remove(animation)
                # If all deal animations complete, check for dealer ace and offer insurance
                if game_state == "DEALING" and not card_animations:
                    # Check if dealer's face-up card is an Ace
                    if dealer_hand[0][0] == 'A':
                        game_state = "INSURANCE"
                    else:
                        game_state = "PLAYER_TURN"

                    # Check for natural blackjack on first hand
                    player_hand = player_hands[0]
                    if calculate_hand(player_hand) == 21:
                        # Check if dealer also has blackjack
                        if calculate_hand(dealer_hand) == 21:
                            game_state = "GAME_OVER"
                            result = "PUSH"
                            show_dealer_cards = True
                            text_effects.append(TextEffect(
                                result,
                                (WIDTH//2, HEIGHT//2),
                                BLUE
                            ))

                            # Process insurance if taken
                            if insurance_bet > 0:
                                # Insurance pays 2:1
                                player_money += insurance_bet * 3
                                stats["insurance_wins"] += 1
                                text_effects.append(TextEffect(
                                    "Insurance Win!",
                                    (WIDTH//2, HEIGHT//2 + 50),
                                    GREEN
                                ))
                        else:
                            game_state = "GAME_OVER"
                            result = "BLACKJACK!"
                            nat_blackjack = pygame.mixer.Sound("assets/sounds/nat_blackjack.wav")
                            nat_blackjack.play()
                            show_dealer_cards = True
                            # Blackjack pays 3:2
                            player_money += int(current_bet * 2.5)
                            text_effects.append(TextEffect(
                                result,
                                (WIDTH//2, HEIGHT//2),
                                GOLD
                            ))

                        particle_systems.extend(check_achievements(
                            game_state, result, player_hand, dealer_hand,
                            player_money, current_bet, achievements_unlocked,
                            text_effects, particle_systems, stats
                        ) or [])

        for animation in chip_animations[:]:
            animation.update()
            if animation.complete:
                chip_animations.remove(animation)

        for effect in text_effects[:]:
            effect.update()
            if effect.complete:
                text_effects.remove(effect)

        for system in particle_systems[:]:
            system.update()
            if system.complete:
                particle_systems.remove(system)

        # Handle dealer turn logic
        if game_state == "DEALER_TURN" and not card_animations:
            # Calculate dealer's hand value
            dealer_value = calculate_hand(dealer_hand)

            # Dealer hits on 16 or less, stands on 17 or more
            if dealer_value < 17:
                dealer_card = deal_card(deck)
                dealer_hand.append(dealer_card)

                # Calculate position based on number of cards
                offset = len(dealer_hand) - 3
                card_animations.append(create_deal_animation(
                    dealer_card,
                    (WIDTH - CARD_WIDTH - 50, 50),
                    (WIDTH//2 + CARD_WIDTH + offset * 30, HEIGHT//2 - 150)
                ))
            else:
                # Process insurance if dealer has blackjack
                if calculate_hand(dealer_hand) == 21 and len(dealer_hand) == 2:
                    if insurance_bet > 0:
                        # Insurance pays 2:1
                        player_money += insurance_bet * 3
                        stats["insurance_wins"] += 1
                        text_effects.append(TextEffect(
                            "Insurance Win!",
                            (WIDTH//2, HEIGHT//2 - 100),
                            GREEN
                        ))

                # Determine winner for each hand
                game_state = "GAME_OVER"

                # If there are split hands, evaluate each one
                if len(player_hands) > 1:
                    total_winnings = 0

                    for i, hand in enumerate(player_hands):
                        hand_result = split_results[i] if i < len(split_results) else "UNKNOWN"

                        # Skip already busted hands
                        if hand_result == "BUST!":
                            # Already lost the bet
                            continue

                        player_value = calculate_hand(hand)

                        # Handle double down (bet is doubled)
                        hand_bet = current_bet * 2 if hand_result == "DOUBLE" else current_bet

                        if player_value > 21:
                        # Player busts, player loses
                            player_lose = pygame.mixer.Sound("assets/sounds/player_lose.wav")
                            player_lose.play()
                            player_money = int(player_money - hand_bet)
                            split_results[i] = "LOSE"
                        elif dealer_value > 21 and player_value <= 21:
                            # Dealer busts, player wins
                            player_win = pygame.mixer.Sound("assets/sounds/player_win.wav")
                            player_win.play()
                            total_winnings += hand_bet * 2  # Return bet + winnings
                            split_results[i] = "WIN"
                        elif dealer_value > player_value:
                            # Dealer wins
                            player_lose = pygame.mixer.Sound("assets/sounds/player_lose.wav")
                            player_lose.play()
                            player_money = int(player_money - hand_bet)
                            split_results[i] = "LOSE"
                        elif dealer_value < player_value and player_value <= 21:
                            # Player wins
                            player_win = pygame.mixer.Sound("assets/sounds/player_win.wav")
                            player_win.play()
                            total_winnings += hand_bet * 2  # Return bet + winnings
                            split_results[i] = "WIN"
                        else:
                            # Push (tie)
                            push = pygame.mixer.Sound("assets/sounds/push.wav")
                            push.play()
                            total_winnings += hand_bet  # Return bet
                            split_results[i] = "PUSH"

                    # Add winnings to player's money
                    player_money += total_winnings

                    # Set overall result
                    if dealer_value > 21:
                        result = "DEALER BUSTS!"
                    else:
                        result = "HANDS SETTLED"

                    text_effects.append(TextEffect(
                        result,
                        (WIDTH//2, HEIGHT//2 - 100),
                        GREEN if dealer_value > 21 else WHITE
                    ))
                else:
                    # Single hand evaluation
                    player_value = calculate_hand(player_hands[0])

                    if dealer_value > 21:
                        result = "DEALER BUSTS!"
                        dealer_busts = pygame.mixer.Sound("assets/sounds/dealer_bust.wav")
                        dealer_busts.play()
                        player_money += current_bet * 2  # Return bet + winnings
                        text_effects.append(TextEffect(
                            result,
                            (WIDTH//2, HEIGHT//2),
                            GREEN
                        ))
                        win.play()
                    elif dealer_value > player_value:
                        result = "DEALER WINS!"
                        dealer_wins = pygame.mixer.Sound("assets/sounds/dealer_win.wav")
                        dealer_wins.play()
                        player_money = int(player_money-current_bet)
                        text_effects.append(TextEffect(
                            result,
                            (WIDTH//2, HEIGHT//2),
                            RED
                        ))
                        lose.play()
                    elif dealer_value < player_value:
                        result = "YOU WIN!"
                        player_wins = pygame.mixer.Sound("assets/sounds/player_win.wav")
                        player_wins.play()
                        player_money += current_bet * 2  # Return bet + winnings
                        text_effects.append(TextEffect(
                            result,
                            (WIDTH//2, HEIGHT//2),
                            GREEN
                        ))
                        win.play()
                    else:
                        result = "PUSH"
                        player_money += current_bet  # Return bet
                        push = pygame.mixer.Sound("assets/sounds/push.wav")
                        push.play()
                        text_effects.append(TextEffect(
                            result,
                            (WIDTH//2, HEIGHT//2),
                            BLUE
                        ))

                particle_systems.extend(check_achievements(
                    game_state, result, player_hands[0], dealer_hand,
                    player_money, current_bet, achievements_unlocked,
                    text_effects, particle_systems, stats
                ) or [])

        # Draw game
        screen.fill(BLACK)

        # Draw casino table background
        screen.blit(CASINO_TABLE, (0, 0))

        # Draw deck
        screen.blit(CARD_BACK, (WIDTH - CARD_WIDTH - 50, 50))

        # Draw dealer's cards
        if dealer_hand:
            for i, card in enumerate(dealer_hand):
                if i == 1 and not show_dealer_cards:
                    # Draw face down card
                    screen.blit(CARD_BACK, (WIDTH//2 + 10, HEIGHT//2 - 150))
                else:
                    # Draw face up card
                    screen.blit(card_images[card], 
                                (WIDTH//2 - CARD_WIDTH - 10 + i * (CARD_WIDTH + 20), HEIGHT//2 - 150))

            # Draw dealer's hand value
            if show_dealer_cards:
                value_text = FONT.render(f"Dealer: {calculate_hand(dealer_hand)}", True, WHITE)
                screen.blit(value_text, (WIDTH//2 - 60, HEIGHT//2 - 200))

        for hand_index, hand in enumerate(player_hands):
            if hand:
                y_offset = hand_index * 100  # Offset for split hands

                for i, card in enumerate(hand):
                    screen.blit(card_images[card], 
                                (WIDTH//2 - CARD_WIDTH - 10 + i * (CARD_WIDTH + 20), HEIGHT//2 + 50 + y_offset))

                # Draw hand value - FIXED POSITION
                # Move the text to the left side of the cards to avoid overlap
                value_text = FONT.render(f"Hand {hand_index + 1}: {calculate_hand(hand)}", True, WHITE)
                screen.blit(value_text, (WIDTH//2 - 250, HEIGHT//2 + 80 + y_offset))  # Changed from HEIGHT//2 + 160 + y_offset

                # For split hands in game over state, show result
                if game_state == "GAME_OVER" and len(player_hands) > 1 and hand_index < len(split_results):
                    result_text = FONT.render(split_results[hand_index], True, 
                                             GREEN if split_results[hand_index] == "WIN" else 
                                             RED if split_results[hand_index] == "LOSE" or split_results[hand_index] == "BUST!" else 
                                             BLUE)
                    # Also move the result text to be adjacent to the hand value
                    screen.blit(result_text, (WIDTH//2 - 100, HEIGHT//2 + 80 + y_offset))  # Changed from WIDTH//2 + 100, HEIGHT//2 + 160 + y_offset

        # Draw money and bet information
        money_text = FONT.render(f"Money: ${player_money}", True, GOLD)
        screen.blit(money_text, (50, 50))

        bet_text = FONT.render(f"Current Bet: ${current_bet}", True, WHITE)
        screen.blit(bet_text, (50, 100))

        # Draw insurance bet if active
        if insurance_bet > 0:
            insurance_text = FONT.render(f"Insurance: ${insurance_bet}", True, WHITE)
            screen.blit(insurance_text, (50, 150))

        # Draw buttons based on game state
        if game_state == "BETTING":
            # Draw chips for betting
            draw_chips(screen, current_bet)

            # Draw deal button if bet is valid
            if current_bet >= MIN_BET:
                draw_glowing_button(screen, deal_button, "DEAL", WHITE, GREEN, (100, 255, 100))

            # Reset button to clear bet
            if current_bet > 0:
                reset_button = pygame.Rect(WIDTH//2 - 200, HEIGHT - 100, 100, 50)
                draw_glowing_button(screen, reset_button, "RESET", WHITE, RED, (255, 100, 100))

                # Handle reset button click
                if pygame.mouse.get_pressed()[0]:
                    mouse_pos = pygame.mouse.get_pos()
                    if reset_button.collidepoint(mouse_pos):
                        current_bet = 0

        # Draw insurance option buttons
        elif game_state == "INSURANCE":
            # Draw message about insurance
            insurance_msg = FONT.render("Dealer showing an Ace. Would you like to take insurance?", True, WHITE)
            screen.blit(insurance_msg, (WIDTH//2 - insurance_msg.get_width()//2, HEIGHT//2))

            insurance_cost = FONT.render(f"Insurance costs: ${current_bet // 2}", True, GOLD)
            screen.blit(insurance_cost, (WIDTH//2 - insurance_cost.get_width()//2, HEIGHT//2 + 40))

            # Draw insurance buttons
            draw_glowing_button(screen, insurance_button, "TAKE INSURANCE", WHITE, GREEN, (100, 255, 100))
            draw_glowing_button(screen, no_insurance_button, "NO INSURANCE", WHITE, RED, (255, 100, 100))

        elif game_state == "PLAYER_TURN":
            # Get current hand
            player_hand = player_hands[current_hand_index]

            # Draw indicator for current hand
            if len(player_hands) > 1:
                indicator_text = FONT.render(f"Playing Hand {current_hand_index + 1}", True, GOLD)
                screen.blit(indicator_text, (WIDTH//2 - indicator_text.get_width()//2, HEIGHT//2))

            # Draw hit and stand buttons
            draw_glowing_button(screen, hit_button, "HIT", WHITE, BLUE, (100, 100, 255))
            draw_glowing_button(screen, stand_button, "STAND", WHITE, RED, (255, 100, 100))

            # Draw double button only for first decision with enough money
            if len(player_hand) == 2 and player_money >= current_bet:
                draw_glowing_button(screen, double_button, "DOUBLE", WHITE, GOLD, (255, 215, 0))

            # Draw split button only if first two cards are the same value
            if len(player_hand) == 2 and player_money >= current_bet:
                card1_value = get_card_value(player_hand[0])
                card2_value = get_card_value(player_hand[1])

                if card1_value == card2_value:
                    draw_glowing_button(screen, split_button, "SPLIT", WHITE, PURPLE, (200, 100, 255))

        elif game_state == "GAME_OVER":
            # Draw new hand button
            draw_glowing_button(screen, hit_button, "NEW HAND", WHITE, GREEN, (100, 100, 225))

            # Draw main menu button
            main_menu_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 50, 200, 40)
            draw_glowing_button(screen, main_menu_button, "MAIN MENU", WHITE, BLUE, (100, 100, 255))

            # Handle main menu button click
            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if main_menu_button.collidepoint(mouse_pos):
                    return  # Exit the game loop and return to the main menu

            # Draw result text
            result_font = pygame.font.Font(None, 72)
            result_text = result_font.render(result, True, GOLD if "WIN" in result or "BLACKJACK" in result else RED if "BUST" in result or "DEALER WINS" in result else BLUE)
            screen.blit(result_text, (WIDTH//2 - result_text.get_width()//2, HEIGHT//2 - 50))

            # Draw winnings/losses text for single hands
            if len(player_hands) == 1:
                if "WIN" in result or "BLACKJACK" in result or "DEALER BUSTS" in result:
                    winnings = int(current_bet * 1.5) if "BLACKJACK" in result else current_bet
                    winnings_text = FONT.render(f"+ ${winnings}", True, GREEN)
                    screen.blit(winnings_text, (WIDTH//2 - winnings_text.get_width()//2, HEIGHT//2 + 20))
                elif "PUSH" in result:
                    push_text = FONT.render("Bet Returned", True, BLUE)
                    screen.blit(push_text, (WIDTH//2 - push_text.get_width()//2, HEIGHT//2 + 20))
                else:
                    loss_text = FONT.render(f"- ${current_bet}", True, RED)
                    screen.blit(loss_text, (WIDTH//2 - loss_text.get_width()//2, HEIGHT//2 + 20))

        # Draw current animations on top of everything
        for animation in card_animations:
            card = animation.card
            if animation.complete:
                continue

    # Check if this is the dealer's face down card
            is_dealer_face_down = False
            if card in dealer_hand and dealer_hand.index(card) == 1 and not show_dealer_cards:
                is_dealer_face_down = True

            if is_dealer_face_down:
                # For the dealer's face-down card, use CARD_BACK for both front and back
                animation.draw(screen, card_images[card], CARD_BACK)
            else:
                # For other cards, use the card's front image and CARD_BACK as the back image
                animation.draw(screen, card_images[card], CARD_BACK)

        # Draw chip animations
        for animation in chip_animations:
            if animation.complete:
                continue

            chip_img = chip_images[animation.value]
            screen.blit(chip_img, animation.current_pos)

        # Draw text effects
        for effect in text_effects:
            effect.draw(screen)

        # Draw particle systems
        for system in particle_systems:
            system.draw(screen)

        # Draw achievements notification if any were unlocked this hand
        if achievements_unlocked:
            # Create achievements area
            achievements_area = pygame.Rect(WIDTH - 300, 150, 250, 100 + len(achievements_unlocked) * 40)
            pygame.draw.rect(screen, (0, 0, 0, 180), achievements_area, border_radius=10)
            pygame.draw.rect(screen, GOLD, achievements_area, 2, border_radius=10)

            # Draw header
            header_font = pygame.font.Font(None, 28)
            header_text = header_font.render("Recent Achievements", True, GOLD)
            screen.blit(header_text, (WIDTH - 300 + (250 - header_text.get_width())//2, 160))

            # Draw achievement names
            for i, key in enumerate(achievements_unlocked):
                achievement_font = pygame.font.Font(None, 24)
                ach_text = achievement_font.render(ACHIEVEMENTS[key]["name"], True, WHITE)
                screen.blit(ach_text, (WIDTH - 280, 200 + i * 40))

        # In the main game loop, add these checks:
        if not deck:
            deck = shuffle_deck()  # Reshuffle if the deck is empty

        if player_money <= 0:
            over = pygame.mixer.Sound("assets/sounds/game_over.wav")
            over.play()
            running = False  # End the game

        if current_bet < 0:
            current_bet = 0  # Prevent negative bets

        # Display FPS
        fps_text = FONT.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        screen.blit(fps_text, (WIDTH - 150, 20))

        # Update the display
        pygame.display.flip()
        clock.tick(60)

# Quit pygame
    pygame.quit()

# Run the game
if __name__ == "__main__":
    # Execute game
    while True:
        # Show the main menu and get the player's choice
        option = main_menu()

        if option == "PLAY":
            # Start a new game
            main()
        elif option == "ACHIEVEMENTS":
            # Show the achievements screen
            show_achievements_screen()
        else:  # QUIT
            # Exit the game
            break

    pygame.quit()
