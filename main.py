import pygame
import random
import os
import time
import math
from pygame import gfxdraw

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
CARD_WIDTH, CARD_HEIGHT = 71, 96
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
GOLD = (255, 215, 0)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
FONT = pygame.font.Font(None, 36)
CHIP_VALUES = [10, 50, 100, 500]  # Chip denominations
MIN_BET, MAX_BET = 10, 500  # Betting limits

# Animation classes
class CardAnimation:
    def __init__(self, card, start_pos, end_pos, duration=0.5):
        self.card = card
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000  # Convert to milliseconds
        self.complete = False
        self.rotation = 0
        self.scale = 1.0
        self.current_pos = start_pos

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True
            self.current_pos = self.end_pos
            self.rotation = 0
            self.scale = 1.0
            return

        # Easing function (ease-out)
        progress = elapsed / self.duration
        progress = 1 - (1 - progress) ** 2

        # Calculate current position
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress

        # Add some rotation and scaling for effect
        self.rotation = math.sin(progress * math.pi) * 15
        self.scale = 1.0 + math.sin(progress * math.pi) * 0.2

        self.current_pos = (x, y)

    def draw(self, screen, image):
        if self.complete:
            screen.blit(image, self.current_pos)
            return

        # Create a rotated and scaled version of the card
        rotated_image = pygame.transform.rotozoom(image, self.rotation, self.scale)
        new_rect = rotated_image.get_rect(center=(self.current_pos[0] + CARD_WIDTH/2, self.current_pos[1] + CARD_HEIGHT/2))
        screen.blit(rotated_image, new_rect.topleft)

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

# Load assets
CASINO_TABLE = load_image(os.path.join("assets", "casino_table.png"), (WIDTH, HEIGHT))
CARD_BACK = load_image(os.path.join("assets", "cards", "back.png"), (CARD_WIDTH, CARD_HEIGHT))

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

# Draw glowing button
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
        pygame.draw.rect(glow_surf, (*glow_color, alpha), 
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

# Main game loop
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack")
    clock = pygame.time.Clock()

    # Game state variables
    deck = shuffle_deck()
    player_hand = []
    dealer_hand = []
    player_money = 1000
    current_bet = 0
    game_state = "betting"
    result = ""
    time_to_reset = 0

    # Animation lists
    card_animations = []
    chip_animations = []
    text_effects = []
    particle_systems = []
    deal_in_progress = False

    # Animation positions
    deck_pos = (WIDTH - 120, HEIGHT // 2 - 100)  # Deck position

    # Create a light source for dynamic lighting
    light_pos = (WIDTH // 2, HEIGHT // 2)
    light_radius = 500

    # Sound effects (placeholder)
    try:
        card_sound = pygame.mixer.Sound(os.path.join("assets", "sounds", "card_flip.wav"))
        chip_sound = pygame.mixer.Sound(os.path.join("assets", "sounds", "chip_place.wav"))
        win_sound = pygame.mixer.Sound(os.path.join("assets", "sounds", "win.wav"))
        lose_sound = pygame.mixer.Sound(os.path.join("assets", "sounds", "lose.wav"))
    except:
        # Create silent sounds as fallback
        card_sound = pygame.mixer.Sound(bytes(44100))
        chip_sound = pygame.mixer.Sound(bytes(44100))
        win_sound = pygame.mixer.Sound(bytes(44100))
        lose_sound = pygame.mixer.Sound(bytes(44100))

    running = True
    while running:
        current_time = pygame.time.get_ticks()

        # Update animations
        for anim in card_animations[:]:
            anim.update()
            if anim.complete:
                card_animations.remove(anim)

        for anim in chip_animations[:]:
            anim.update()
            if anim.complete:
                chip_animations.remove(anim)

        for effect in text_effects[:]:
            effect.update()
            if effect.complete:
                text_effects.remove(effect)

        for particles in particle_systems[:]:
            particles.update()
            if particles.complete:
                particle_systems.remove(particles)

        # Check if deal animations are complete
        if deal_in_progress and not card_animations:
            deal_in_progress = False

        # Draw background
        screen.fill(GREEN)
        if CASINO_TABLE:
            screen.blit(CASINO_TABLE, (0, 0))

        # Draw deck of cards
        pygame.draw.rect(screen, (0, 0, 0, 128), (deck_pos[0], deck_pos[1], CARD_WIDTH, CARD_HEIGHT))
        screen.blit(CARD_BACK, (deck_pos[0], deck_pos[1]))

        # Draw dynamic lighting effects (simple version)
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 60))  # Darken the scene

        # Create light circle
        light_radius_dynamic = light_radius + math.sin(current_time / 1000 * 2) * 50
        for r in range(int(light_radius_dynamic), 0, -10):
            alpha = 60 - int(60 * (r / light_radius_dynamic))
            pygame.draw.circle(overlay, (255, 255, 255, alpha), light_pos, r, 10)

        # Apply lighting
        # screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_MULT)

        # Draw player's hand (static cards)
        for i, card in enumerate(player_hand):
            pos = (400 + i * (CARD_WIDTH + 10), HEIGHT - CARD_HEIGHT - 200)
            # Skip drawing if there's an animation for this card
            if not any(anim.card == card and not anim.complete for anim in card_animations):
                screen.blit(card_images[card], pos)

        # Draw dealer's hand (static cards)
        for i, card in enumerate(dealer_hand):
            pos = (400 + i * (CARD_WIDTH + 10), 100)
            # Skip if there's an animation for this card
            if not any(anim.card == card and not anim.complete for anim in card_animations):
                if game_state == "player_turn" and i == 0:
                    screen.blit(CARD_BACK, pos)
                else:
                    screen.blit(card_images[card], pos)

        # Draw card animations
        for anim in card_animations:
            if anim.card in player_hand or anim.card in dealer_hand:
                card_image = card_images[anim.card]
                # For dealer's first card during player turn, show card back
                if game_state == "player_turn" and anim.card == dealer_hand[0]:
                    card_image = CARD_BACK
                anim.draw(screen, card_image)

        # Draw chip animations
        for anim in chip_animations:
            if not anim.complete:
                chip_image = chip_images[anim.value]
                screen.blit(chip_image, anim.current_pos)

        # Draw text effects
        for effect in text_effects:
            effect.draw(screen)

        # Draw particle systems
        for particles in particle_systems:
            particles.draw(screen)

        # Display text
        player_value = calculate_hand(player_hand) if player_hand else 0
        dealer_value = 0
        if dealer_hand:
            if game_state == "player_turn":
                dealer_value = "?"
            else:
                dealer_value = calculate_hand(dealer_hand)

        player_text = FONT.render(f'Player: {player_value}', True, WHITE)
        dealer_text = FONT.render(f'Dealer: {dealer_value}', True, WHITE)
        money_text = FONT.render(f'Money: ${player_money}', True, GOLD)

        # Add text shadows
        player_shadow = FONT.render(f'Player: {player_value}', True, BLACK)
        dealer_shadow = FONT.render(f'Dealer: {dealer_value}', True, BLACK)
        money_shadow = FONT.render(f'Money: ${player_money}', True, BLACK)

        # Draw text with shadows
        screen.blit(player_shadow, (52, HEIGHT - CARD_HEIGHT - 198))
        screen.blit(player_text, (50, HEIGHT - CARD_HEIGHT - 200))
        screen.blit(dealer_shadow, (52, 122))
        screen.blit(dealer_text, (50, 120))
        screen.blit(money_shadow, (WIDTH - 198, 22))
        screen.blit(money_text, (WIDTH - 200, 20))

        # Draw current bet with pulsing effect if betting
        if game_state == "betting" or current_bet > 0:
            pulse = math.sin(current_time / 300) * 0.2 + 0.8
            bet_size = int(36 + pulse * 10) if game_state == "betting" else 36
            bet_font = pygame.font.Font(None, bet_size)
            bet_text = bet_font.render(f"Bet: ${current_bet}", True, GOLD)
            bet_shadow = bet_font.render(f"Bet: ${current_bet}", True, BLACK)

            text_rect = bet_text.get_rect(center=(WIDTH // 2, HEIGHT - 200))
            shadow_rect = bet_shadow.get_rect(center=(WIDTH // 2 + 2, HEIGHT - 198))

            screen.blit(bet_shadow, shadow_rect)
            screen.blit(bet_text, text_rect)

        # Draw chips for betting
        if game_state == "betting":
            draw_chips(screen, current_bet)

        # Draw buttons based on game state
        if game_state == "player_turn" and not deal_in_progress:
            # Hit button with glow
            hit_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 100, 90, 50)
            draw_glowing_button(screen, hit_rect, "Hit", WHITE, (180, 0, 0), (255, 100, 100))

            # Stand button with glow
            stand_rect = pygame.Rect(WIDTH // 2 + 10, HEIGHT - 100, 90, 50)
            draw_glowing_button(screen, stand_rect, "Stand", WHITE, (0, 0, 180), (100, 100, 255))

        elif game_state == "betting" and not deal_in_progress:
            # Deal button if bet is placed
            if current_bet >= MIN_BET:
                deal_rect = pygame.Rect(WIDTH // 2 - 50, HEIGHT - 100, 100, 50)
                draw_glowing_button(screen, deal_rect, "Deal", WHITE, (0, 100, 0), (100, 255, 100), pulse=True)

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and not deal_in_progress:
                x, y = pygame.mouse.get_pos()

                if game_state == "betting":
                    # Chip selection
                    for i, value in enumerate(CHIP_VALUES):
                        chip_x = WIDTH // 2 - 250 + i * 120
                        if chip_x <= x <= chip_x + 60 and HEIGHT - 150 <= y <= HEIGHT - 90:
                            if player_money >= current_bet + value and current_bet + value <= MAX_BET:
                                current_bet += value

                                # Add chip animation
                                start_x, start_y = chip_x, y
                                end_x = WIDTH // 2
                                end_y = HEIGHT - 250
                                chip_animations.append(ChipAnimation(value, (start_x, start_y), (end_x, end_y)))

                                # Play sound
                                chip_sound.play()

                                # Add particles
                                particle_systems.append(ParticleSystem((x, y), GOLD, count=15))

                    # Deal button
                    if current_bet >= MIN_BET and WIDTH // 2 - 50 <= x <= WIDTH // 2 + 50 and HEIGHT - 100 <= y <= HEIGHT - 50:
                        # Start the game with animation
                        player_money -= current_bet
                        deal_in_progress = True
                        game_state = "dealing"

                        # Create new hands
                        player_hand = []
                        dealer_hand = []

                        # Create deal animations with staggered timing
                        # First card to player
                        card = deal_card(deck)
                        player_hand.append(card)
                        card_animations.append(create_deal_animation(
                            card, deck_pos, (400, HEIGHT - CARD_HEIGHT - 200)
                        ))

                        # First card to dealer
                        card = deal_card(deck)
                        dealer_hand.append(card)
                        card_animations.append(create_deal_animation(
                            card, deck_pos, (400, 100)
                        ))

                        # Second card to player
                        card = deal_card(deck)
                        player_hand.append(card)
                        card_animations.append(create_deal_animation(
                            card, deck_pos, (400 + CARD_WIDTH + 10, HEIGHT - CARD_HEIGHT - 200)
                        ))

                        # Second card to dealer
                        card = deal_card(deck)
                        dealer_hand.append(card)
                        card_animations.append(create_deal_animation(
                            card, deck_pos, (400 + CARD_WIDTH + 10, 100)
                        ))

                        # Play card sounds
                        card_sound.play()

                        # Once animations complete, game state will transition to player_turn

                elif game_state == "player_turn":
                    # Hit button
                    if WIDTH // 2 - 100 <= x <= WIDTH // 2 - 10 and HEIGHT - 100 <= y <= HEIGHT - 50:
                        # Add new card with animation
                        card = deal_card(deck)
                        player_hand.append(card)

                        # Card animation
                        pos_x = 400 + len(player_hand) * (CARD_WIDTH + 10) - (CARD_WIDTH + 10)
                        card_animations.append(create_deal_animation(
                            card, deck_pos, (pos_x, HEIGHT - CARD_HEIGHT - 200)
                        ))

                        # Play card sound
                        card_sound.play()

                        # Check for bust after animation completes
                        if calculate_hand(player_hand) > 21:
                            game_state = "game_over"
                            result = "BUST!"
                            text_effects.append(TextEffect(
                                "BUSTED!", 
                                (WIDTH // 2, HEIGHT // 2), 
                                RED
                            ))
                            lose_sound.play()
                            time_to_reset = current_time + 2000

                    # Stand button
                    elif WIDTH // 2 + 10 <= x <= WIDTH // 2 + 100 and HEIGHT - 100 <= y <= HEIGHT - 50:
                        game_state = "dealer_turn"

        # If all animations complete and game state is dealing, move to player turn
        if game_state == "dealing" and not card_animations:
            game_state = "player_turn"

        # Handle dealer turn
        if game_state == "dealer_turn" and not deal_in_progress:
            # Dealer draws cards until at least 17
            if calculate_hand(dealer_hand) < 17:
                # Add new card with animation
                card = deal_card(deck)
                dealer_hand.append(card)

                deal_in_progress = True

                # Card animation
                pos_x = 400 + len(dealer_hand) * (CARD_WIDTH + 10) - (CARD_WIDTH + 10)
                card_animations.append(create_deal_animation(
                    card, deck_pos, (pos_x, 100)
                ))

                # Play card sound
                card_sound.play()
            else:
                game_state = "game_over"
                player_value = calculate_hand(player_hand)
                dealer_value = calculate_hand(dealer_hand)

                if dealer_value > 21:
                    result = "DEALER BUSTS!"
                    player_money += current_bet * 2
                    text_effects.append(TextEffect(
                        "YOU WIN!", 
                        (WIDTH // 2, HEIGHT // 2), 
                        GOLD, 
                        size_start=40, 
                        size_end=80
                    ))
                    particle_systems.append(ParticleSystem((WIDTH // 2, HEIGHT // 2), GOLD, count=50, duration=2.0))
                    win_sound.play()
                elif player_value > dealer_value:
                    result = "YOU WIN!"
                    player_money += current_bet * 2
                    text_effects.append(TextEffect(
                        "YOU WIN!", 
                        (WIDTH // 2, HEIGHT // 2), 
                        GOLD, 
                        size_start=40, 
                        size_end=80
                    ))
                    particle_systems.append(ParticleSystem((WIDTH // 2, HEIGHT // 2), GOLD, count=50, duration=2.0))
                    win_sound.play()
                elif player_value < dealer_value:
                    result = "DEALER WINS"
                    text_effects.append(TextEffect(
                        "DEALER WINS", 
                        (WIDTH // 2, HEIGHT // 2), 
                        RED
                    ))
                    lose_sound.play()
                else:
                    result = "PUSH"
                    player_money += current_bet
                    text_effects.append(TextEffect(
                        "PUSH - TIE GAME", 
                        (WIDTH // 2, HEIGHT // 2), 
                        BLUE
                    ))
                time_to_reset = current_time + 3000

        # Display result
        if game_state == "game_over" and result:
            result_text = FONT.render(result, True, WHITE)
            result_shadow = FONT.render(result, True, BLACK)
            result_rect = result_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100))

            screen.blit(result_shadow, (result_rect.x + 2, result_rect.y + 2))
            screen.blit(result_text, result_rect)

            # Show a "Continue" button after the animations are done
            if current_time >= time_to_reset and not text_effects:
                cont_rect = pygame.Rect(WIDTH // 2 - 75, HEIGHT // 2 + 50, 150, 50)
                draw_glowing_button(screen, cont_rect, "Continue", WHITE, (0, 100, 0), (100, 255, 100), pulse=True)

                # Check for button click
                if pygame.mouse.get_pressed()[0]:
                    x, y = pygame.mouse.get_pos()
                    if cont_rect.collidepoint(x, y):
                        game_state = "betting"
                        current_bet = 0
                        result = ""

                        # Check for game over
                        if player_money < MIN_BET:
                            # Game over - not enough money
                            text_effects.append(TextEffect(
                                "GAME OVER - OUT OF MONEY!", 
                                (WIDTH // 2, HEIGHT // 2), 
                                RED,
                                duration=4.0
                            ))
                            # Reset player money after game over message
                            time_to_reset = current_time + 4000

        # Check if player is out of money and needs to reset
        if game_state == "betting" and player_money < MIN_BET and current_time >= time_to_reset:
            player_money = 1000  # Reset player money
            text_effects.append(TextEffect(
                "NEW GAME - $1000 ADDED", 
                (WIDTH // 2, HEIGHT // 2), 
                GOLD,
                duration=2.0
            ))

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()