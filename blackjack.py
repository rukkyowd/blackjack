import os
import sys
import subprocess
import importlib

def check_and_install_packages():
    required_packages = [
        'pygame',
        'numpy',
        'scikit-learn',
        'matplotlib' 
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"{package} is already installed")
        except ImportError:
            print(f"{package} not found. Installing...")
            try:
                # Use pip to install the package
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please install it manually.")


# Check and install packages before anything else
check_and_install_packages()

import pygame
import random
import time
import math
import pygame.gfxdraw
import logging
import collections
import numpy as np # type: ignore
from collections import deque
from sklearn.linear_model import SGDRegressor # type: ignore
from typing import Dict, List, Any, Callable, Tuple

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
GRAY = (128, 128, 128)
STAND_COLOR = (0, 150, 0)      # Green for stand
HIT_COLOR = (200, 0, 0)        # Red for hit
DOUBLE_COLOR = (200, 150, 0)   # Orange/gold for double down
SPLIT_COLOR = (0, 200, 200)   # Blue for split
SURRENDER_COLOR = (150, 150, 150)  # Gray for surrender
FONT = pygame.font.Font(None, 36)
CHIP_VALUES = [10, 50, 100, 250, 500, 1000]
MIN_BET, MAX_BET = 10, 1000000000000
INSURANCE_BUTTON = pygame.Rect(WIDTH // 2 - 75, HEIGHT - 200, 150, 50)
SPLIT_BUTTON = pygame.Rect(WIDTH // 2 + 200, HEIGHT - 100, 100, 50)
STRATEGY_WIDTH = 330
STRATEGY_HEIGHT = 500
STRATEGY_MARGIN = 20
CELL_SIZE = 30
JACKPOT_ENABLED = False  # Whether the jackpot is active
JACKPOT_AMOUNT = 0  # Current jackpot amount
JACKPOT_CONTRIBUTION = 0.01  # 1% of each bet contributes to the jackpot
JACKPOT_TRIGGER = "natural_blackjack"  # Condition to win the jackpot
PLAYER_MONEY = 1000  # Initial money
VIP_ROOM_ACTIVE = False # Whether the VIP Room is active

def save_player_money(player_money):
    try:
        with open("player_money.txt", "w") as file:
            file.write(str(player_money))
        logging.info(f"Player money saved: {player_money}")
    except Exception as e:
        logging.error(f"Error saving player money: {e}")

def load_player_money():
    try:
        with open("player_money.txt", "r") as file:
            money = int(file.read())
            logging.info(f"Player money loaded: {money}")
            return money
    except FileNotFoundError:
        logging.warning("Player money file not found, using default value.")
        return 1000  # Default starting money if file doesn't exist
    except Exception as e:
        logging.error(f"Error loading player money: {e}")
        return 1000  # Default starting money if there's an error

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
win.set_volume(0.3)
lose.set_volume(0.3)
CASINO_TABLE = load_image(os.path.join("assets", "casino_table.png"),(WIDTH, HEIGHT))
CARD_BACK = load_image(os.path.join("assets", "cards", "back.png"),(CARD_WIDTH, CARD_HEIGHT))
VIP_CASINO_TABLE = load_image(os.path.join("assets", "vip_casino_table.jpg"), (WIDTH, HEIGHT))
VIP_CHIP_IMAGES = {v: load_image(f"assets/chips/vip_chip_{v}.png") for v in CHIP_VALUES}

#Logging Console

log_file = "console_output.log"
open(log_file, "w").close()  # Manually clear the file

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),  # Overwrite file
        logging.StreamHandler(sys.stdout)  # Print to console
    ])

import random
import numpy as np
from typing import List, Tuple, Callable, Dict, Any

class RiskAssessmentModel:
    def __init__(self):
        """
        Advanced risk assessment with comprehensive probabilistic modeling
        and adaptive learning capabilities for blackjack strategy optimization.
        """
        # Expanded probabilistic risk factors with confidence intervals
        self.risk_factors = {
            'hand_composition': {
                'pair': {'mean': 1.2, 'std': 0.1},
                'soft_hand': {'mean': 0.9, 'std': 0.05},
                'hard_hand': {'mean': 1.1, 'std': 0.08},
                'blackjack': {'mean': 0.4, 'std': 0.02},  # Very low risk for blackjack
                'stiff_hand': {'mean': 1.4, 'std': 0.12}  # 12-16 hard total
            },
            'dealer_upcard': {
                'weak': {'mean': 0.7, 'std': 0.05},    # 2-6
                'neutral': {'mean': 1.0, 'std': 0.05}, # 7-9
                'strong': {'mean': 1.3, 'std': 0.1}    # 10, Ace
            },
            'hand_total': {
                'low': {'mean': 0.8, 'std': 0.05},     # Less than 12
                'medium': {'mean': 1.2, 'std': 0.1},    # 12-16
                'high': {'mean': 0.9, 'std': 0.08}      # 17+
            },
            'deck_composition': {
                'high_cards_rich': {'mean': 0.85, 'std': 0.07},  # Lots of 10s, faces, aces
                'low_cards_rich': {'mean': 1.15, 'std': 0.07}    # Lots of 2-6
            }
        }
        
        # Bayesian priors for risk factor adaptation
        self.factor_priors = {factor: 1.0 for factor in self.risk_factors}
        
        # Enhanced learning parameters
        self.learning_rate = 0.05
        self.decay_factor = 0.98
        self.min_learning_rate = 0.01
        
        # Decision history for adaptive learning
        self.decision_history = []
        self.max_history_size = 100
        
        # Simulation parameters
        self.simulation_trials = 1000
        
        # Initialize variance reduction techniques
        self.use_control_variates = True
        self.variance_baseline = {}

    def calculate_dealer_bust_probability(self, dealer_value: int) -> float:
        """
        Calculate the probability of the dealer busting based on their upcard
        using Monte Carlo simulation with variance reduction.
        
        Args:
            dealer_value: Value of dealer's upcard (1-11)
            
        Returns:
            Probability (0-1) that dealer will bust
        """
        # Enhanced dealer bust probabilities based on upcard
        bust_probabilities = {
            1: 0.17,   # Ace
            2: 0.35,
            3: 0.37,
            4: 0.40,
            5: 0.42,
            6: 0.42,
            7: 0.26,
            8: 0.24,
            9: 0.23,
            10: 0.23,  # 10, J, Q, K
            11: 0.17   # Ace (same as 1)
        }
        
        # Get base probability
        base_prob = bust_probabilities.get(min(dealer_value, 11), 0.23)
        
        # Add more sophisticated variance with confidence bound
        rand_factor = np.random.normal(1.0, 0.03)
        bounded_factor = max(0.95, min(1.05, rand_factor))
        
        return base_prob * bounded_factor
    
    def get_hand_type(self, player_hand: List[Tuple], get_card_value: Callable) -> Dict[str, bool]:
        """
        Analyze the hand type for more detailed risk assessment.
        
        Args:
            player_hand: Current player hand
            get_card_value: Function to get card value
            
        Returns:
            Dictionary of hand characteristics
        """
        hand_types = {
            'is_pair': False,
            'is_soft': False,
            'is_stiff': False,
            'is_blackjack': False,
            'total': 0
        }
        
        # Check if hand is a pair
        if len(player_hand) == 2 and player_hand[0][0] == player_hand[1][0]:
            hand_types['is_pair'] = True
        
        # Calculate hand total and check if soft
        total = 0
        ace_count = 0
        
        for card in player_hand:
            value = get_card_value(card)
            if value == 1:  # Ace
                ace_count += 1
                total += 11
            else:
                total += value
        
        # Adjust for aces if needed
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1
        
        hand_types['total'] = total
        hand_types['is_soft'] = (ace_count > 0)
        
        # Check if blackjack
        if len(player_hand) == 2 and total == 21:
            hand_types['is_blackjack'] = True
        
        # Check if stiff hand (hard 12-16)
        if 12 <= total <= 16 and not hand_types['is_soft']:
            hand_types['is_stiff'] = True
            
        return hand_types
    
    def calculate_comprehensive_risk(
        self, 
        player_hand: List[Tuple], 
        dealer_upcard: Tuple, 
        true_count: float, 
        get_card_value: Callable,
        remaining_decks: float = 1.0
    ) -> Dict[str, Any]:
        """
        Enhanced probabilistic multi-dimensional risk assessment with
        confidence intervals and Monte Carlo simulation.
        
        Args:
            player_hand: Current player hand
            dealer_upcard: Dealer's visible card
            true_count: Current card counting true count
            get_card_value: Function to get card value
            remaining_decks: Number of decks remaining
        
        Returns:
            Dictionary with comprehensive risk analysis
        """
        # Sample from risk factor distributions with confidence intervals
        def sample_risk_factor(factor_config):
            return max(0.5, min(2.0, np.random.normal(
                factor_config['mean'], 
                factor_config['std']
            )))
        
        # Initialize risk components for detailed analysis
        risk_components = {}
        base_risk = 1.0
        
        # Get hand characteristics
        hand_info = self.get_hand_type(player_hand, get_card_value)
        hand_total = hand_info['total']
        
        # Monte Carlo simulations for bust probability if hitting
        bust_prob = self.simulate_hit_outcome(hand_total, true_count, remaining_decks)
        risk_components['bust_probability'] = bust_prob
        
        # Apply hand composition risk factors
        if hand_info['is_blackjack']:
            risk_factor = sample_risk_factor(self.risk_factors['hand_composition']['blackjack'])
            base_risk *= risk_factor
            risk_components['blackjack_factor'] = risk_factor
        elif hand_info['is_pair']:
            risk_factor = sample_risk_factor(self.risk_factors['hand_composition']['pair'])
            base_risk *= risk_factor
            risk_components['pair_factor'] = risk_factor
        elif hand_info['is_soft']:
            risk_factor = sample_risk_factor(self.risk_factors['hand_composition']['soft_hand'])
            base_risk *= risk_factor
            risk_components['soft_hand_factor'] = risk_factor
        elif hand_info['is_stiff']:
            risk_factor = sample_risk_factor(self.risk_factors['hand_composition']['stiff_hand'])
            base_risk *= risk_factor
            risk_components['stiff_hand_factor'] = risk_factor
        else:
            risk_factor = sample_risk_factor(self.risk_factors['hand_composition']['hard_hand'])
            base_risk *= risk_factor
            risk_components['hard_hand_factor'] = risk_factor
        
        # Apply hand total risk assessment
        if hand_total < 12:
            risk_factor = sample_risk_factor(self.risk_factors['hand_total']['low'])
            base_risk *= risk_factor
            risk_components['low_total_factor'] = risk_factor
        elif hand_total <= 16:
            risk_factor = sample_risk_factor(self.risk_factors['hand_total']['medium'])
            base_risk *= risk_factor
            risk_components['medium_total_factor'] = risk_factor
        else:
            risk_factor = sample_risk_factor(self.risk_factors['hand_total']['high'])
            base_risk *= risk_factor
            risk_components['high_total_factor'] = risk_factor
        
        # Dealer upcard risk assessment
        dealer_val = get_card_value(dealer_upcard)
        if dealer_val in [2, 3, 4, 5, 6]:
            risk_factor = sample_risk_factor(self.risk_factors['dealer_upcard']['weak'])
            base_risk *= risk_factor
            risk_components['dealer_weak_factor'] = risk_factor
            
            # Calculate dealer bust probability
            dealer_bust_prob = self.calculate_dealer_bust_probability(dealer_val)
            risk_components['dealer_bust_probability'] = dealer_bust_prob
            
            # Adjust risk based on dealer bust probability
            base_risk *= (2.0 - dealer_bust_prob)
            
        elif dealer_val in [10, 1]:  # 10 or Ace
            risk_factor = sample_risk_factor(self.risk_factors['dealer_upcard']['strong'])
            base_risk *= risk_factor
            risk_components['dealer_strong_factor'] = risk_factor
        else:
            risk_factor = sample_risk_factor(self.risk_factors['dealer_upcard']['neutral'])
            base_risk *= risk_factor
            risk_components['dealer_neutral_factor'] = risk_factor
        
        # Card counting influence with dynamic adaptation
        count_influence = 0
        if true_count > 0:
            # Positive count favors the player - reduce risk
            count_influence = -0.05 * true_count
        else:
            # Negative count favors the dealer - increase risk
            count_influence = -0.05 * true_count  # Will be positive as true_count is negative
        
        # Apply deck composition risk factors based on true count
        if true_count > 2:
            risk_factor = sample_risk_factor(self.risk_factors['deck_composition']['high_cards_rich'])
            base_risk *= risk_factor
            risk_components['high_cards_rich_factor'] = risk_factor
        elif true_count < -2:
            risk_factor = sample_risk_factor(self.risk_factors['deck_composition']['low_cards_rich'])
            base_risk *= risk_factor
            risk_components['low_cards_rich_factor'] = risk_factor
            
        # Apply count influence
        count_factor = 1 + count_influence
        base_risk *= count_factor
        risk_components['count_factor'] = count_factor
        
        # Calculate expected value based on risk assessment
        ev_estimation = self.estimate_expected_value(hand_info, dealer_val, true_count, base_risk)
        risk_components['estimated_ev'] = ev_estimation
        
        # Add small variance for final risk score
        final_risk = max(0.5, min(
            base_risk * np.random.normal(1, 0.02),  # Reduced variance
            2.0
        ))
        
        return {
            'risk_score': final_risk,
            'components': risk_components,
            'hand_info': hand_info,
            'recommendation': self.generate_recommendation(final_risk, hand_info, dealer_val)
        }
    
    def simulate_hit_outcome(self, current_total: int, true_count: float, remaining_decks: float) -> float:
        """
        Simulate the probability of busting if hitting
        
        Args:
            current_total: Current hand total
            true_count: Current true count
            remaining_decks: Number of decks remaining
            
        Returns:
            Probability of busting if hitting
        """
        # If already over 21, 100% bust probability
        if current_total >= 21:
            return 1.0 if current_total > 21 else 0.0
            
        # Calculate how many card values would cause a bust
        bust_threshold = 21 - current_total
        
        # Standard distribution of card values
        card_distribution = {
            1: 4,   # Aces
            2: 4,
            3: 4,
            4: 4,
            5: 4,
            6: 4,
            7: 4,
            8: 4,
            9: 4,
            10: 16  # 10, J, Q, K
        }
        
        # Adjust for true count (approximating effect on deck composition)
        if true_count != 0:
            # Positive count means more low cards have been seen
            high_cards_adjustment = 0.01 * true_count * sum(card_distribution.values())
            
            # Adjust 10-value cards and low cards based on count
            for card_val in card_distribution:
                if card_val >= 10:
                    # More high cards remain with positive count
                    card_distribution[card_val] += high_cards_adjustment
                elif card_val <= 6:
                    # Fewer low cards remain with positive count
                    card_distribution[card_val] -= high_cards_adjustment / 5  # Split among 5 low cards
        
        # Ensure all probabilities are positive
        for card_val in card_distribution:
            card_distribution[card_val] = max(0, card_distribution[card_val])
        
        # Calculate total cards and bust probability
        total_cards = sum(card_distribution.values()) * remaining_decks
        bust_cards = sum(card_distribution.get(i, 0) for i in range(bust_threshold + 1, 11)) * remaining_decks
        
        return bust_cards / total_cards if total_cards > 0 else 0.5
    
    def estimate_expected_value(
        self, 
        hand_info: Dict[str, Any], 
        dealer_val: int, 
        true_count: float,
        risk_score: float
    ) -> float:
        """
        Estimate the expected value of the current hand
        
        Args:
            hand_info: Information about the player's hand
            dealer_val: Dealer's upcard value
            true_count: Current true count
            risk_score: Calculated risk score
            
        Returns:
            Estimated expected value (-1 to 1)
        """
        # Base expected value calculations
        # Higher risk generally correlates with negative EV
        base_ev = 2.0 - risk_score * 2.0  # Maps risk 0.5->1.0, 1.0->0, 2.0->-2.0
        
        # Adjust for hand strength
        if hand_info['is_blackjack']:
            base_ev += 1.0
        elif hand_info['total'] >= 20:
            base_ev += 0.5
        elif hand_info['total'] <= 12:
            base_ev -= 0.2
            
        # Adjust for dealer strength
        if dealer_val in [2, 3, 4, 5, 6]:
            base_ev += 0.3
        elif dealer_val in [10, 1]:  # 10 or Ace
            base_ev -= 0.3
            
        # Adjust for true count
        base_ev += 0.05 * true_count
        
        # Constrain to reasonable EV range
        return max(-1.0, min(1.0, base_ev))
    
    def generate_recommendation(
        self, 
        risk_score: float, 
        hand_info: Dict[str, Any], 
        dealer_val: int
    ) -> str:
        """
        Generate a strategy recommendation based on risk assessment
        
        Args:
            risk_score: Calculated risk score
            hand_info: Information about the player's hand
            dealer_val: Dealer's upcard value
            
        Returns:
            Strategy recommendation (hit, stand, double, split)
        """
        if hand_info['is_blackjack']:
            return "stand"
            
        if hand_info['total'] > 21:
            return "bust"
            
        if hand_info['is_pair']:
            # Pair splitting logic
            card_val = hand_info['total'] // 2
            if card_val in [8, 11]:  # 8s or Aces
                return "split"
            elif card_val == 9 and dealer_val not in [7, 10, 1]:
                return "split"
            elif card_val in [2, 3, 7] and dealer_val <= 7:
                return "split"
            elif card_val == 6 and dealer_val <= 6:
                return "split"
            elif card_val == 4 and dealer_val in [5, 6]:
                return "split"
                
        if hand_info['is_soft']:
            # Soft hand logic
            if hand_info['total'] >= 19:
                return "stand"
            elif hand_info['total'] == 18:
                if dealer_val in [9, 10, 1]:
                    return "hit"
                elif dealer_val in [2, 3, 4, 5, 6] and len(hand_info) == 2:
                    return "double" 
                else:
                    return "stand"
            elif hand_info['total'] == 17:
                if dealer_val in [3, 4, 5, 6] and len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
            elif hand_info['total'] in [15, 16]:
                if dealer_val in [4, 5, 6] and len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
            elif hand_info['total'] in [13, 14]:
                if dealer_val in [5, 6] and len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
        else:
            # Hard hand logic
            if hand_info['total'] >= 17:
                return "stand"
            elif hand_info['total'] in [13, 14, 15, 16]:
                if dealer_val <= 6:
                    return "stand"
                else:
                    return "hit"
            elif hand_info['total'] == 12:
                if dealer_val in [4, 5, 6]:
                    return "stand"
                else:
                    return "hit"
            elif hand_info['total'] == 11:
                if len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
            elif hand_info['total'] == 10:
                if dealer_val <= 9 and len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
            elif hand_info['total'] == 9:
                if dealer_val in [3, 4, 5, 6] and len(hand_info) == 2:
                    return "double"
                else:
                    return "hit"
                    
        # Default to hit for low values
        return "hit"
    
    def update_risk_factors(self, outcome: Dict[str, Any]) -> None:
        """
        Update risk factors based on outcome of previous decisions
        
        Args:
            outcome: Dictionary with decision outcomes
        """
        # Store decision in history
        self.decision_history.append(outcome)
        if len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)
        
        # Extract relevant information
        decision = outcome.get('decision')
        result = outcome.get('result')  # win, loss, push
        risk_score = outcome.get('risk_score')
        risk_components = outcome.get('components', {})
        
        # Only update if we have valid outcome data
        if not all([decision, result, risk_score]):
            return
            
        # Calculate adaptation direction
        adaptation_direction = 0
        if result == 'win' and risk_score > 1.0:
            # We won despite high risk assessment - reduce risk factors
            adaptation_direction = -1
        elif result == 'loss' and risk_score < 1.0:
            # We lost despite low risk assessment - increase risk factors
            adaptation_direction = 1
            
        # Apply updates to affected risk factors
        for component_name, factor_value in risk_components.items():
            if 'factor' not in component_name:
                continue
                
            # Extract category and type
            parts = component_name.split('_')
            if len(parts) < 2:
                continue
                
            # Try to identify risk factor category and type
            category = None
            factor_type = None
            
            for cat in self.risk_factors:
                for ftype in self.risk_factors[cat]:
                    if ftype in component_name:
                        category = cat
                        factor_type = ftype
                        break
                if category:
                    break
            
            # Apply adaptation if factor was identified
            if category and factor_type:
                current_mean = self.risk_factors[category][factor_type]['mean']
                adjustment = self.learning_rate * adaptation_direction
                new_mean = max(0.4, min(1.6, current_mean + adjustment))
                self.risk_factors[category][factor_type]['mean'] = new_mean
        
        # Decay learning rate over time for stability
        self.learning_rate = max(self.min_learning_rate, 
                                self.learning_rate * self.decay_factor)
                                
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze historical performance and risk factor effectiveness
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.decision_history:
            return {"status": "No historical data available"}
            
        # Calculate win rate
        wins = sum(1 for outcome in self.decision_history if outcome.get('result') == 'win')
        losses = sum(1 for outcome in self.decision_history if outcome.get('result') == 'loss')
        pushes = sum(1 for outcome in self.decision_history if outcome.get('result') == 'push')
        
        total_decisions = len(self.decision_history)
        win_rate = wins / total_decisions if total_decisions > 0 else 0
        
        # Calculate risk assessment accuracy
        correct_assessments = 0
        for outcome in self.decision_history:
            risk_score = outcome.get('risk_score', 1.0)
            result = outcome.get('result')
            
            # Low risk should correlate with wins, high risk with losses
            if (risk_score < 1.0 and result == 'win') or (risk_score > 1.0 and result == 'loss'):
                correct_assessments += 1
                
        accuracy = correct_assessments / total_decisions if total_decisions > 0 else 0
        
        return {
            "total_decisions": total_decisions,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "risk_assessment_accuracy": accuracy,
            "learning_rate": self.learning_rate,
            "risk_factors": self.risk_factors
        }
    
class PerformanceAnalytics:
    def __init__(self, strategy_modes):
        self.session_metrics = {
            'total_hands': 0,
            'net_winnings': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'current_win_streak': 0,
            'current_loss_streak': 0
        }
        
        self.strategy_performance = {
            mode: {
                'hands_played': 0,
                'net_result': 0,
                'win_rate': 0.0,
                'average_bet': 0.0,
                'average_decision_accuracy': 0.0
            } for mode in strategy_modes.keys()
        }
    
    def record_hand_result(self, strategy_mode, result, bet_amount, decision_accuracy=None):
        """
        Record detailed performance for each hand
        
        Args:
        - strategy_mode: Current strategy mode used
        - result: Hand result (positive for win, negative for loss)
        - bet_amount: Amount bet on the hand
        - decision_accuracy: Accuracy of the decision made (optional)
        """
        perf = self.strategy_performance[strategy_mode]
        perf['hands_played'] += 1
        perf['net_result'] += result
        
        # Update average bet
        perf['average_bet'] = (
            perf['average_bet'] * (perf['hands_played'] - 1) + bet_amount
        ) / perf['hands_played']
        
        # Update win rate (simplified)
        perf['win_rate'] = (perf['net_result'] > 0) * 100
        
        # Update decision accuracy if provided
        if decision_accuracy is not None:
            current_total_accuracy = perf['average_decision_accuracy'] * (perf['hands_played'] - 1)
            perf['average_decision_accuracy'] = (
                current_total_accuracy + decision_accuracy
            ) / perf['hands_played']
        
        # Update session metrics
        self.session_metrics['total_hands'] += 1
        self.session_metrics['net_winnings'] += result
        
        # Update win/loss streaks
        if result > 0:
            self.session_metrics['current_win_streak'] += 1
            self.session_metrics['current_loss_streak'] = 0
            self.session_metrics['max_win_streak'] = max(
                self.session_metrics['max_win_streak'], 
                self.session_metrics['current_win_streak']
            )
        elif result < 0:
            self.session_metrics['current_loss_streak'] += 1
            self.session_metrics['current_win_streak'] = 0
            self.session_metrics['max_loss_streak'] = max(
                self.session_metrics['max_loss_streak'], 
                self.session_metrics['current_loss_streak']
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
        "description":
        "Get a natural Blackjack (21 with your first two cards).",
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

def calculate_hand(hand):
    value = sum(card_values[card[0]] for card in hand)
    aces = sum(1 for card in hand if card[0] == 'ace')
    while value > 21 and aces:
        value -= 10
        aces -= 1
    return value

def dealer_should_hit(dealer_hand):
    """Standard dealer rules: hit on 16 or less, stand on 17 or more."""
    dealer_value = calculate_hand(dealer_hand)
    return dealer_value <= 16

def get_suit_color(suit):
    """Returns 'red' or 'black' for a given suit"""
    return "red" if suit in ["hearts", "diamonds"] else "black"

# Animation classes
import pygame
import math
import random
from typing import Tuple, Optional

class CardAnimation:
    def __init__(self,
                 card,
                 start_pos: Tuple[float, float],
                 end_pos: Tuple[float, float],
                 duration: float = 0.5,
                 flip_duration: float = 0.3,
                 ease_type: str = "out_quad",
                 with_shadow: bool = True,
                 arc_height: float = 50.0):
        """
        Enhanced card animation with easing, arcs, and better visual effects.
        
        Args:
            card: The card object being animated
            start_pos: Starting position (x, y)
            end_pos: Target position (x, y)
            duration: Total animation duration in seconds
            flip_duration: Flip animation duration in seconds
            ease_type: Easing function type ('linear', 'in_quad', 'out_quad', etc.)
            with_shadow: Whether to render a shadow effect
            arc_height: Height of the arc trajectory (0 for straight line)
        """
        self.card = card
        self.start_pos = pygame.Vector2(start_pos)
        self.end_pos = pygame.Vector2(end_pos)
        self.control_point = self._calculate_control_point(arc_height)
        self.start_time = pygame.time.get_ticks()
        self.duration = duration * 1000  # Convert to milliseconds
        self.flip_duration = flip_duration * 1000
        self.complete = False
        self.flip_progress = 0  # 0 to 1
        self.is_flipping = False
        self.current_pos = pygame.Vector2(start_pos)
        self.current_rotation = 0
        self.current_scale = 1.0
        self.ease_type = ease_type
        self.with_shadow = with_shadow
        self.shadow_offset = 5
        self.shadow_alpha = 100
        self.arc_height = arc_height
        self.z_index = 0  # For draw order management
        
        # Physics-like properties
        self.velocity = pygame.Vector2(0, 0)
        self.angular_velocity = random.uniform(-2, 2)
        self.damping = 0.98
        
        # Visual effects
        self.glow_radius = 0
        self.glow_alpha = 0
        self.glow_color = (255, 255, 200)

    def _calculate_control_point(self, arc_height: float) -> pygame.Vector2:
        """Calculate control point for quadratic bezier curve."""
        mid_point = (self.start_pos + self.end_pos) / 2
        direction = (self.end_pos - self.start_pos).normalize()
        perpendicular = pygame.Vector2(-direction.y, direction.x)
        return mid_point + perpendicular * arc_height

    def _ease(self, t: float, ease_type: str) -> float:
        """Easing functions for smooth animations."""
        if ease_type == "linear":
            return t
        elif ease_type == "in_quad":
            return t * t
        elif ease_type == "out_quad":
            return t * (2 - t)
        elif ease_type == "in_out_quad":
            if t < 0.5:
                return 2 * t * t
            else:
                return -1 + (4 - 2 * t) * t
        elif ease_type == "in_back":
            s = 1.70158
            return t * t * ((s + 1) * t - s)
        elif ease_type == "out_back":
            s = 1.70158
            t = t - 1
            return t * t * ((s + 1) * t + s) + 1
        else:
            return t  # default to linear

    def update(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.complete = True
            self.current_pos = self.end_pos.copy()
            self.velocity = pygame.Vector2(0, 0)
            self.current_rotation = 0
            return

        # Calculate normalized progress with easing
        progress = self._ease(elapsed / self.duration, self.ease_type)

        # Quadratic bezier curve for arc movement
        self.current_pos = (
            (1 - progress) ** 2 * self.start_pos +
            2 * (1 - progress) * progress * self.control_point +
            progress ** 2 * self.end_pos
        )

        # Calculate velocity for physics-like movement
        if elapsed > 0:
            new_pos = self.current_pos.copy()
            self.velocity = (new_pos - self.current_pos) * (1000 / (elapsed + 1))
            self.current_pos = new_pos

        # Rotation based on movement direction
        if self.velocity.length() > 0.1:
            target_angle = math.degrees(math.atan2(self.velocity.y, self.velocity.x))
            self.current_rotation = pygame.math.lerp(
                self.current_rotation, 
                target_angle * 0.1, 
                0.1
            )

        # Damping for physics effects
        self.velocity *= self.damping
        self.angular_velocity *= self.damping

        # Handle flip animation
        if elapsed < self.flip_duration:
            self.is_flipping = True
            self.flip_progress = elapsed / self.flip_duration
            # Add slight scale effect during flip
            self.current_scale = 1.0 + 0.1 * math.sin(self.flip_progress * math.pi)
        else:
            self.is_flipping = False
            self.current_scale = 1.0

        # Glow effect when nearly complete
        if progress > 0.8:
            self.glow_radius = int(20 * (progress - 0.8) / 0.2)
            self.glow_alpha = int(100 * (progress - 0.8) / 0.2)
        else:
            self.glow_radius = 0
            self.glow_alpha = 0

    def draw(self, screen: pygame.Surface, front_image: pygame.Surface, back_image: pygame.Surface):
        if self.complete:
            self._draw_card(screen, front_image, self.end_pos)
            return

        if self.is_flipping:
            self._draw_flip_animation(screen, front_image, back_image)
        else:
            self._draw_card(screen, front_image, self.current_pos)

    def _draw_card(self, screen: pygame.Surface, image: pygame.Surface, pos: pygame.Vector2):
        """Draw the card with optional shadow and rotation."""
        # Create a temporary surface for transformations
        card_rect = pygame.Rect(0, 0, image.get_width(), image.get_height())
        transformed_image = pygame.Surface(card_rect.size, pygame.SRCALPHA)
        
        # Apply rotation and scale
        if self.current_rotation != 0 or self.current_scale != 1.0:
            # Scale first
            scaled_size = (int(card_rect.width * self.current_scale), 
                          int(card_rect.height * self.current_scale))
            scaled_image = pygame.transform.scale(image, scaled_size)
            # Then rotate
            rotated_image = pygame.transform.rotate(scaled_image, self.current_rotation)
            transformed_image = rotated_image
        else:
            transformed_image = image.copy()

        # Draw shadow
        if self.with_shadow:
            shadow_offset = self.shadow_offset * (1 - (self.current_pos.y / screen.get_height()))
            shadow_rect = transformed_image.get_rect(center=pos + pygame.Vector2(shadow_offset, shadow_offset))
            shadow_surface = pygame.Surface(transformed_image.get_size(), pygame.SRCALPHA)
            shadow_surface.fill((0, 0, 0, self.shadow_alpha))
            screen.blit(shadow_surface, shadow_rect)

        # Draw glow
        if self.glow_radius > 0:
            glow_surface = pygame.Surface((self.glow_radius*2, self.glow_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*self.glow_color, self.glow_alpha), 
                              (self.glow_radius, self.glow_radius), self.glow_radius)
            screen.blit(glow_surface, pos - pygame.Vector2(self.glow_radius, self.glow_radius))

        # Draw card
        card_rect = transformed_image.get_rect(center=pos)
        screen.blit(transformed_image, card_rect)

    def _draw_flip_animation(self, screen: pygame.Surface, front_image: pygame.Surface, back_image: pygame.Surface):
        """Draw the card flip animation with perspective effects."""
        pos = self.current_pos
        width, height = front_image.get_size()
        
        # Calculate flip progress with easing
        flip_t = self._ease(self.flip_progress, "out_quad")
        
        # Shadow during flip
        if self.with_shadow:
            shadow_size = (int(width * (1 - abs(flip_t - 0.5) * 0.5)), 
                         int(height * 0.9))
            shadow_offset = self.shadow_offset * (1 - (pos.y / screen.get_height()))
            shadow_pos = (pos.x - shadow_size[0]//2 + shadow_offset, 
                         pos.y - shadow_size[1]//2 + shadow_offset)
            shadow_surface = pygame.Surface(shadow_size, pygame.SRCALPHA)
            shadow_surface.fill((0, 0, 0, self.shadow_alpha))
            screen.blit(shadow_surface, shadow_pos)

        # Draw the appropriate side based on flip progress
        if flip_t < 0.5:
            # First half: show back image with perspective
            scale_x = 1.0 - flip_t * 2
            scale_y = 1.0 - abs(flip_t - 0.25) * 0.5  # Slight vertical squash
            scaled_image = pygame.transform.scale(
                back_image, 
                (int(width * scale_x), int(height * scale_y))
            )
            screen.blit(scaled_image,
                      (pos.x - scaled_image.get_width() // 2,
                       pos.y - scaled_image.get_height() // 2))
        else:
            # Second half: show front image with perspective
            scale_x = (flip_t - 0.5) * 2
            scale_y = 1.0 - abs(flip_t - 0.75) * 0.5  # Slight vertical squash
            scaled_image = pygame.transform.scale(
                front_image, 
                (int(width * scale_x), int(height * scale_y))
            )
            screen.blit(scaled_image,
                      (pos.x - scaled_image.get_width() // 2,
                       pos.y - scaled_image.get_height() // 2))
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
        x = self.start_pos[0] + (self.end_pos[0] -
                                 self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] -
                                 self.start_pos[1]) * progress

        # Add arc effect
        y -= math.sin(progress * math.pi) * 100

        self.current_pos = (x, y)


class TextEffect:

    def __init__(self,
                 text,
                 position,
                 color,
                 duration=2.0,
                 size_start=36,
                 size_end=66):
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
            self.current_size = self.size_start + (
                self.size_end - self.size_start) * size_progress
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

class AchievementNotification:
    def __init__(self, achievement_key):
        self.key = achievement_key
        self.start_time = pygame.time.get_ticks()
        self.duration = 5000  # 5 seconds
        self.position = (WIDTH//2, 50)
        self.particles = ParticleSystem(self.position, GOLD)
        
    def draw(self, screen):
        elapsed = pygame.time.get_ticks() - self.start_time
        if elapsed > self.duration:
            return False
            
        # Draw achievement card
        card_rect = pygame.Rect(0, 0, 600, 100)
        card_rect.center = self.position
        pygame.draw.rect(screen, (30, 30, 60), card_rect, border_radius=15)
        
        # Draw text
        title = FONT.render(ACHIEVEMENTS[self.key]['name'], True, GOLD)
        desc = FONT.render(ACHIEVEMENTS[self.key]['description'], True, WHITE)
        screen.blit(title, (card_rect.x + 20, card_rect.y + 15))
        screen.blit(desc, (card_rect.x + 20, card_rect.y + 50))
        
        self.particles.draw(screen)
        return True

class StrategyAssistant:
    def __init__(self, strategy_mode='balanced'):
        self._setup_logging()  
        # Font initialization with error handling
        try:
            self.font = pygame.font.Font(None, 20)
            self.small_font = pygame.font.Font(None, 16)  
            self.title_font = pygame.font.Font(None, 24)
        except pygame.error:
            print("Warning: Unable to initialize fonts. Ensure pygame is initialized.")
            self.font = None
            self.small_font = None
            self.title_font = None

        self.deviations = {}

        # Original betting strategy before mode adjustments
        self.original_betting_strategy = [
            (-10, 1),
            (-5, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 4),
            (4, 6),
            (5, 8),
            (float('inf'), 10)
        ]

        self.screen_width = 800  # Use the global WIDTH constant
        self.screen_height = 600  # Use the global HEIGHT constant
        self.panel_width = STRATEGY_WIDTH  # Use the global STRATEGY_WIDTH constant
        self.panel_height = STRATEGY_HEIGHT  # Use the global STRATEGY_HEIGHT constant
        self.min_panel_width = 300  # Minimum width in pixels
        self.min_panel_height = 400  # Minimum height in pixels
        self.active = False
        self.surface = pygame.Surface((STRATEGY_WIDTH, STRATEGY_HEIGHT), pygame.SRCALPHA)
        self.vip_mode = False 

        self.actions = ['HIT', 'STAND', 'DOUBLE', 'SPLIT', 'SURRENDER']
        
        # Collapsible sections state
        self.sections = {
            'heatmap': True,
            'probability': True,
            'advanced_stats': False,
        }

        # Animation state
        self.animation_state = {
            'mode_switch': 0,  # 0-100 for animation progress
            'ev_meter': 0,     # Current EV value for animation
            'target_ev': 0,    # Target EV value to animate to
            'bust_prob': 0,   # Current bust probability
            'target_bust_prob': 0,
            'confetti': [],    # Active confetti particles
            'pulse_phase': 0,
            'section_slide': {section: 0 for section in self.sections}
        }

        # Enhanced visual themes
        self.visual_themes = {
            'classic': {
                'background': (20, 30, 40),
                'primary': (45, 65, 90),
                'highlight': (255, 215, 0),
                'text_primary': (240, 240, 240),
                'text_secondary': (180, 180, 200),
                'action_colors': {
                    'HIT': (100, 200, 100),
                    'STAND': (200, 100, 100),
                    'DOUBLE': (100, 100, 200),
                    'SPLIT': (100, 200, 200)
                }
            },
            'dark_mode': {
                'background': (10, 10, 20),
                'primary': (30, 40, 60),
                'highlight': (200, 165, 0),
                'text_primary': (220, 220, 230),
                'text_secondary': (150, 150, 170),
                'action_colors': {
                    'HIT': (80, 180, 80),
                    'STAND': (180, 80, 80),
                    'DOUBLE': (80, 80, 180),
                    'SPLIT': (80, 180, 180)
                }
            },
            'high_contrast': {
                'background': (0, 0, 0),
                'primary': (50, 50, 50),
                'highlight': (255, 255, 0),
                'text_primary': (255, 255, 255),
                'text_secondary': (200, 200, 200),
                'action_colors': {
                    'HIT': (0, 255, 0),
                    'STAND': (255, 0, 0),
                    'DOUBLE': (0, 0, 255),
                    'SPLIT': (0, 255, 255)
                }
            }
        }
        
        # Current theme
        self.current_theme = 'classic'
        self.color_palette = self.visual_themes[self.current_theme].copy()
        
        # Enhanced visualization parameters
        self.visualization_options = {
            'show_probability_overlay': True,
            'show_expected_value': True,
            'display_card_count_impact': True
        }
        
        # Interaction state parameters
        self.interaction_state = {
            'hover_effects': True,
            'animation_speed': 1.0,
            'tooltips_enabled': True
        }
        
        # Tooltip system
        self._tooltips = {
            'strategy_mode': "Switch between conservative, balanced, and aggressive strategies",
            'probability_overlay': "Shows your chances of busting and expected value",
            'decision_history': "Green = Correct decision, Red = Suboptimal decision",
            'action_legend': "Expected value of each action in current conditions",
            'count_info': "Running and true count based on Hi-Lo"
        }
        
        # Expanded risk assessment parameters
        self.risk_thresholds = {
            'conservative': 0.3,
            'balanced': 0.5,
            'aggressive': 0.7
        }
        
        # Card counting variables
        self.running_count = 0
        self.true_count = 0
        self.deck_penetration = 0
        self.cards_seen = 0
        self.total_cards = 8 * 52  # Assuming 8 decks
        self.count_system = {
            '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
            '7': 0, '8': 0, '9': 0,
            '10': -1, 'jack': -1, 'queen': -1, 'king': -1, 'ace': -1
        }
        
        # Strategy mode configuration
        self.strategy_modes = self._initialize_strategy_modes()
        self.current_mode = None
        self.set_strategy_mode(strategy_mode)
        
        # Initialize base strategy tables
        self._initialize_base_strategies()
        
        # Deviation tables (indexed by true count)
        self.deviations = {
            (16, 10): (0, 'S'),
            (15, 10): (4, 'S'),
            (13, 2): (-1, 'H'),
            (13, 3): (-2, 'H'),
            (12, 4): (3, 'S'),
            (12, 5): (-1, 'S'),
            (12, 6): (-1, 'S'),
            (11, 11): (1, 'D'),
            (10, 10): (4, 'D'),
            (9, 2): (1, 'D'),
            (9, 7): (3, 'D'),
            (8, 6): (2, 'D'),
            ('10,10', 5): (5, 'Y'),
            ('10,10', 6): (4, 'Y'),
            ('A,A', 5): (-1, 'Y'),
            ('A,A', 6): (-1, 'Y'),
            ('9,9', 7): (3, 'Y'),
            ('9,9', 10): (1, 'Y'),
            ('7,7', 10): (4, 'Y'),
            ('6,6', 2): (1, 'Y'),
            ('4,4', 4): (3, 'Y'),
            ('3,3', 8): (1, 'Y'),
            ('2,2', 3): (1, 'Y'),
        }
        
        # Enhanced counting system
        self.advanced_counting = {
            'composition_dependent': True,
            'ace_side_count': 0,
            'ten_side_count': 0
        }
        
        # More granular deviation points
        self.enhanced_deviations = {
            # Additional contextual deviations
            'dealer_bust_probability': {
                2: 0.4,   # 40% chance of busting
                3: 0.4,
                4: 0.4,
                5: 0.4,
                6: 0.4,
                7: 0.3,
                8: 0.2,
                9: 0.2,
                10: 0.1,
                1: 0.2    # Soft 17 rule consideration
            },
            'player_bust_threshold': {
                'hard': 0.5,
                'soft': 0.3
            }
        }
        
        # Set initial strategy mode
        self.current_mode = None
        self.set_strategy_mode(strategy_mode)
        
        # Tracking and statistics
        self.decision_history = []
        self.decision_stats = {
            'HIT': self._create_action_stats(),
            'STAND': self._create_action_stats(),
            'DOUBLE': self._create_action_stats(),
            'SPLIT': self._create_action_stats()
        }
        
        # Visual enhancements
        self.pulse_alpha = 0
        self.pulse_direction = 1
        self.last_recommendation = ""
        self.last_decision_correct = None
        self.animation_counter = 0
        
        # Initialize risk assessment
        self.risk_assessment = RiskAssessmentModel()
        self.current_bet = 0  # Track current bet amount
        
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging for strategy interactions and performance"""
        self.logger = logging.getLogger('StrategyAssistant')
        self.logger.setLevel(logging.INFO)
        
        # Create console and file handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('strategy_assistant.log')
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def update_screen_dimensions(self, width, height):
        """Update dimensions when screen size changes"""
        self.screen_width = width
        self.screen_height = height
        self._update_panel_dimensions()
        
    def _update_panel_dimensions(self):
        """Calculate panel dimensions based on screen size with better proportions"""
        # Use 30% of screen width but no less than 300 and no more than 400
        self.panel_width = min(400, max(300, int(self.screen_width * 0.3)))

        # Use 60% of screen height but no less than 500 and no more than 700
        self.panel_height = min(700, max(500, int(self.screen_height * 0.6)))
    
        # Recreate surface with new dimensions
        self.surface = pygame.Surface((self.panel_width, self.panel_height), pygame.SRCALPHA)
        
    def toggle_section(self, section_name):
        """Toggle collapsible sections"""
        if section_name in self.sections:
            self.sections[section_name] = not self.sections[section_name]
            self.animation_state['section_slide'][section_name] = 100 if self.sections[section_name] else 0
            
    def draw_collapsible_section(self, surface: pygame.Surface, title: str, content_func: callable, 
                               x: int, y: int, width: int, is_open: bool) -> int:
        """Draw a collapsible section with title and content"""
        # Animate section opening/closing
        if is_open and self.animation_state['section_slide'][title.lower()] < 100:
            self.animation_state['section_slide'][title.lower()] += 10
        elif not is_open and self.animation_state['section_slide'][title.lower()] > 0:
            self.animation_state['section_slide'][title.lower()] -= 10
        
        # Draw header
        header_height = 30
        header_rect = pygame.Rect(x, y, width, header_height)
        
        # Hover effect
        mouse_pos = pygame.mouse.get_pos()
        if header_rect.collidepoint(mouse_pos) and self.interaction_state['hover_effects']:
            pygame.draw.rect(surface, self._adjust_color(self.color_palette['primary'], header_rect))
        else:
            pygame.draw.rect(surface, self.color_palette['primary'], header_rect)
            
        # Draw header text
        arrow = "▼" if is_open else "▶"
        title_text = self.font.render(f"{arrow} {title}", True, self.color_palette['text_primary'])
        surface.blit(title_text, (x + 10, y + 5))
        
        # Draw content if open (with animation)
        content_height = 0
        if is_open or self.animation_state['section_slide'][title.lower()] > 0:
            # Calculate animation progress (0-1)
            anim_progress = self.animation_state['section_slide'][title.lower()] / 100
            
            # Draw content with clipping to animate height
            content_y = y + header_height + 5
            clip_rect = pygame.Rect(x, content_y, width, int(500 * anim_progress))  # Assume max 500px height
            old_clip = surface.get_clip()
            surface.set_clip(clip_rect)
            
            # Get content height from content function
            content_height = content_func(surface, x, content_y, width)
            
            surface.set_clip(old_clip)
            
        return header_height + (int(content_height * anim_progress) if anim_progress > 0 else 0)
    
    def _adjust_color(self, color: Tuple[int, int, int], amount: int = 30) -> Tuple[int, int, int]:
        """Lighten or darken a color by the specified amount"""
        return (
            max(0, min(255, color[0] + amount)),
            max(0, min(255, color[1] + amount)),
            max(0, min(255, color[2] + amount))
        )

    def _animate_mode_switch(self):
        """Animate the strategy mode switch"""
        if self.animation_state['mode_switch'] > 0:
            self.animation_state['mode_switch'] -= 2
            return True
        return False
        
    def _animate_ev_meter(self):
        """Smoothly animate EV meter to target value"""
        current = self.animation_state['ev_meter']
        target = self.animation_state['target_ev']
        
        if abs(current - target) < 0.01:
            self.animation_state['ev_meter'] = target
            return False
            
        self.animation_state['ev_meter'] += (target - current) * 0.1
        return True
        
    def _animate_bust_prob(self):
        """Smoothly animate bust probability meter"""
        current = self.animation_state['bust_prob']
        target = self.animation_state['target_bust_prob']
        
        if abs(current - target) < 0.01:
            self.animation_state['bust_prob'] = target
            return False
            
        self.animation_state['bust_prob'] += (target - current) * 0.1
        return True
        
    def _update_confetti(self):
        """Update confetti particle positions"""
        for particle in self.animation_state['confetti']:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.1  # Gravity
            particle['life'] -= 1
            
        # Remove dead particles
        self.animation_state['confetti'] = [p for p in self.animation_state['confetti'] if p['life'] > 0]
        
    def trigger_confetti(self, x, y, count=50):
        """Create confetti explosion at position"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        for _ in range(count):
            self.animation_state['confetti'].append({
                'x': x,
                'y': y,
                'vx': random.uniform(-3, 3),
                'vy': random.uniform(-10, -5),
                'color': random.choice(colors),
                'size': random.randint(3, 8),
                'life': random.randint(30, 60)
            })
            
    def draw_ev_meter(self, surface: pygame.Surface, x: int, y: int, width: int, height: int):
        """Draw animated EV meter with enhanced visualization"""
        # Background
        pygame.draw.rect(surface, self.color_palette['background'], (x, y, width, height))
        
        # Meter track with 3D effect
        pygame.draw.rect(surface, (50, 50, 50), (x, y, width, height))
        pygame.draw.line(surface, (100, 100, 100), (x, y), (x + width, y), 1)
        pygame.draw.line(surface, (30, 30, 30), (x, y + height), (x + width, y + height), 1)
        
        # Current value with animation
        ev_value = self.animation_state['ev_meter']
        meter_width = int(width * max(0, min(1, ev_value)))
        
        # Color gradient based on value
        if ev_value < 0.3:
            color = (255, 0, 0)  # Red
        elif ev_value < 0.6:
            color = (255, 255, 0)  # Yellow
        else:
            color = (0, 255, 0)  # Green
            
        pygame.draw.rect(surface, color, (x, y, meter_width, height))
        
        # Text label with shadow for better visibility
        label_text = f"EV: {ev_value:.2f}"
        label_shadow = self.font.render(label_text, True, (0, 0, 0))
        label = self.font.render(label_text, True, (255, 255, 255))
        
        text_x = x + width//2 - label.get_width()//2
        text_y = y + height//2 - label.get_height()//2
        
        surface.blit(label_shadow, (text_x+1, text_y+1))
        surface.blit(label, (text_x, text_y))
        
        # Add ticks and markers
        for i in range(0, 11):
            tick_x = x + int(width * (i / 10))
            pygame.draw.line(surface, (200, 200, 200), (tick_x, y + height), (tick_x, y + height - 5), 1)
            
            if i % 2 == 0:
                tick_label = self.small_font.render(str(i/10), True, self.color_palette['text_primary'])
                surface.blit(tick_label, (tick_x - tick_label.get_width()//2, y + height + 2))
        
    def draw_bust_prob_gauge(self, surface: pygame.Surface, x: int, y: int, radius: int):
        """Draw speedometer-style bust probability gauge with enhanced features"""
        # Draw gauge background with gradient
        for angle in range(180, 0, -1):
            rad_angle = math.radians(angle)
            end_x = x + radius * math.cos(rad_angle)
            end_y = y - radius * math.sin(rad_angle)
            
            # Color gradient from green to red
            ratio = angle / 180
            color = (
                int(255 * ratio),
                int(255 * (1 - ratio)),
                0
            )
            pygame.draw.line(surface, color, (x, y), (end_x, end_y), 3)
        
        # Draw gauge outline
        pygame.draw.arc(surface, (100, 100, 100), 
                       (x - radius, y - radius, radius*2, radius*2),
                       math.pi, 0, 3)
        
        # Calculate current angle (180° to 0°)
        prob = self.animation_state['bust_prob']
        angle = math.pi - (math.pi * prob)
        
        # Draw indicator with arrowhead
        end_x = x + radius * math.cos(angle)
        end_y = y - radius * math.sin(angle)
        pygame.draw.line(surface, (255, 255, 255), (x, y), (end_x, end_y), 3)
        
        # Draw arrowhead
        arrow_angle1 = angle + math.pi/8
        arrow_angle2 = angle - math.pi/8
        arrow_len = 10
        
        pygame.draw.line(surface, (255, 255, 255), 
                        (end_x, end_y),
                        (end_x + arrow_len * math.cos(arrow_angle1), 
                         end_y - arrow_len * math.sin(arrow_angle1)), 2)
        pygame.draw.line(surface, (255, 255, 255), 
                        (end_x, end_y),
                        (end_x + arrow_len * math.cos(arrow_angle2), 
                         end_y - arrow_len * math.sin(arrow_angle2)), 2)
        
        # Draw labels with percentage markers
        low_label = self.font.render("0%", True, self.color_palette['text_primary'])
        high_label = self.font.render("100%", True, self.color_palette['text_primary'])
        surface.blit(low_label, (x - radius - low_label.get_width(), y - low_label.get_height()//2))
        surface.blit(high_label, (x + radius, y - high_label.get_height()//2))
        
        # Center text with shadow
        prob_text = self.font.render(f"{prob*100:.0f}%", True, (255, 255, 255))
        prob_shadow = self.font.render(f"{prob*100:.0f}%", True, (0, 0, 0))
        surface.blit(prob_shadow, (x - prob_text.get_width()//2 + 1, y + radius//2 + 1))
        surface.blit(prob_text, (x - prob_text.get_width()//2, y + radius//2))
        
    def draw_confetti(self, surface: pygame.Surface):
        """Draw all active confetti particles with enhanced effects"""
        for particle in self.animation_state['confetti']:
            # Draw particle with shadow for depth
            pygame.draw.rect(surface, (30, 30, 30), 
                           (particle['x'] + 1, particle['y'] + 1, 
                            particle['size'], particle['size']))
            pygame.draw.rect(surface, particle['color'], 
                           (particle['x'], particle['y'], 
                            particle['size'], particle['size']))
        
    def draw_confetti(self, surface):
        """Draw all active confetti particles"""
        for particle in self.animation_state['confetti']:
            pygame.draw.rect(surface, particle['color'], 
                           (particle['x'], particle['y'], 
                            particle['size'], particle['size']))
                            
    def update_animations(self):
        """Update all active animations"""
        self._animate_mode_switch()
        self._animate_ev_meter()
        self._animate_bust_prob()
        self._update_confetti()
        
        # Update pulse animation
        self.animation_state['pulse_phase'] = (self.animation_state['pulse_phase'] + 0.05) % (2 * math.pi)
        
    def set_strategy_mode(self, mode='balanced'):
        """Override to trigger animation"""
        if mode != self.current_mode:
            self.animation_state['mode_switch'] = 100  # Start animation
        super().set_strategy_mode(mode)
        
    def log_decision(self, recommended, player_decision, result=0):
        """Override to trigger confetti for correct decisions"""
        decision_correct = player_decision.upper() == recommended.split()[0].upper()
        
        if decision_correct and result > 0:
            # Trigger confetti at recommendation position
            self.trigger_confetti(self.panel_width//2, 150)
            
        super().log_decision(recommended, player_decision, result)
        
    def update_risk_assessment(self, player_hand, dealer_card):
        """Update with animation targets"""
        risk = self.calculate_advanced_risk(player_hand, dealer_card)
        self.animation_state['target_ev'] = risk['expected_value']
        self.animation_state['target_bust_prob'] = risk['bust_probability']
        
    def draw(self, screen: pygame.Surface, player_hand: List[Tuple[str, str]], 
            dealer_up_card: Tuple[str, str], player_decision: str = None):
        """Main draw method with responsive layout and animations"""
        if not self.active:
            return
            
        # Update animations
        self.update_animations()
        
        # Update risk assessment
        if player_hand and dealer_up_card:
            self.update_risk_assessment(player_hand, dealer_up_card)
        
        # Clear surface with semi-transparent background
        self.surface.fill((*self.color_palette['background'], 220))
        
        # Draw sections in a responsive layout
        y_offset = 10
        padding = 10
        
        # Title section with mode indicator
        title_height = self._draw_title_section(y_offset)
        y_offset += title_height + 5

        mode_selector_height = self._draw_mode_selector(y_offset)
        y_offset += mode_selector_height + 10
        
        # Hand information section
        hand_info_height = self._draw_hand_info_section(self.surface, y_offset, player_hand, dealer_up_card)
        y_offset += hand_info_height + padding
        
        # Recommendation section with enhanced animation
        rec_height = self._draw_recommendation_section(y_offset, player_hand, dealer_up_card)
        y_offset += rec_height + padding
        
        # Collapsible probability overlay
        if self.sections['probability']:
            prob_height = self._draw_probability_section(y_offset, player_hand, dealer_up_card)
            y_offset += prob_height + padding
            
        # Collapsible decision history
        if self.sections['heatmap'] and len(self.decision_history) > 0:
            history_height = self.draw_decision_history(self.surface, 20, y_offset, self.panel_width-40)
            y_offset += history_height + padding
            
        # Draw confetti on top of everything
        self.draw_confetti(self.surface)
        
        # Position panel at bottom left with some margin
        panel_x = 20
        panel_y = screen.get_height() - self.panel_height - 20
        screen.blit(self.surface, (panel_x, panel_y))

    def _draw_mode_selector(self, y_offset: int) -> int:
        """Draw the mode selector below the title"""
        section_height = 80
        margin = 20
        width_extension = 200

        panel_x = margin
        panel_y = self.screen_height - section_height - margin
        
        # Section background
        pygame.draw.rect(self.surface, (*self.color_palette['primary'], 150), 
                        (panel_x, panel_y, self.panel_width - margin*2 + width_extension, section_height),
                        border_radius=5)
        
        # Title
        title = self.font.render("Strategy Mode:", True, self.color_palette['text_primary'])
        self.surface.blit(title, (30, y_offset + 10))
        
        # Mode buttons
        modes = self.get_available_modes()
        button_width = 80
        button_height = 30
        spacing = 15
        
        total_width = len(modes) * (button_width + spacing) - spacing
        start_x = (self.panel_width - total_width) // 2
        
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_pos = (mouse_pos[0] - 20, mouse_pos[1] - (self.screen_height - self.panel_height - 40))
        
        for i, mode in enumerate(modes):
            x = start_x + i * (button_width + spacing)
            y = y_offset + 40
            
            mode_rect = pygame.Rect(x, y, button_width, button_height)
            
            # Highlight current mode and hover effects
            if mode == self.current_mode:
                pygame.draw.rect(self.surface, self.color_palette['highlight'], mode_rect, border_radius=5)
            elif mode_rect.collidepoint(adjusted_mouse_pos) and self.interaction_state['hover_effects']:
                pygame.draw.rect(self.surface, self._adjust_color(self.color_palette['primary'], 30), mode_rect, border_radius=5)
            else:
                pygame.draw.rect(self.surface, self.color_palette['primary'], mode_rect, border_radius=5)

            # Mode text
            font = pygame.font.Font(None, 17)
            text = font.render(mode.capitalize(), True, self.color_palette['text_primary'])
            text_rect = text.get_rect(center=mode_rect.center)
            self.surface.blit(text, text_rect)
            
            # Handle clicks
            if mode_rect.collidepoint(adjusted_mouse_pos) and pygame.mouse.get_pressed()[0]:
                self.set_strategy_mode(mode)
        
        return section_height
        
    def _draw_title_section(self, y_offset: int) -> int:
        """Draw title with mode switch animation"""
        # Background with mode color accent
        mode_color = self.strategy_modes[self.current_mode].get('color', (255, 215, 0))
        pygame.draw.rect(self.surface, (*mode_color, 50), (0, 0, self.panel_width, 40))
        
        # Animated title during mode switch
        if self.animation_state['mode_switch'] > 0:
            alpha = 255 * (1 - self.animation_state['mode_switch'] / 100)
            title = self.title_font.render("Blackjack Strategy Advisor", True, (*self.color_palette['text_primary'], alpha))
        else:
            title = self.title_font.render("Blackjack Strategy Advisor", True, self.color_palette['text_primary'])
        
        title_x = self.panel_width//2 - title.get_width()//2
        self.surface.blit(title, (title_x, y_offset))
        
        return 40  # Fixed title height

    def _draw_hand_info_section(self, surface: pygame.Surface, y_offset: int, 
                              player_hand: List[Tuple[str, str]], 
                              dealer_up_card: Tuple[str, str]) -> int:
        """Draw enhanced hand information with card visuals."""
        section_height = 120
        padding = 10

        # Draw section background with rounded corners
        pygame.draw.rect(surface, (*self.color_palette['primary'], 150), 
                         (20, y_offset, self.panel_width - 40, section_height), 
                         border_radius=8)
        pygame.draw.rect(surface, self.color_palette['highlight'], 
                         (20, y_offset, self.panel_width - 40, section_height), 
                         2, border_radius=8)

        if player_hand and dealer_up_card:
            # --- Player Hand ---
            player_value = self.calculate_hand(player_hand)
            is_soft = self.contains_ace(player_hand) and player_value <= 21

            # Player label
            player_label = self.font.render("Your Hand:", True, self.color_palette['text_primary'])
            surface.blit(player_label, (30, y_offset + 15))

            # Player cards (miniature card visuals)
            card_x, card_y = 30, y_offset + 40
            for i, card in enumerate(player_hand[:4]):  # Limit to 4 cards for space
                # Draw card outline
                pygame.draw.rect(surface, WHITE, (card_x + i * 25, card_y, 20, 30), 1, border_radius=2)
                # Draw card value (simplified)
                card_value = self.get_card_value(card)
                value_text = self.small_font.render(
                    "A" if card_value == 1 else str(card_value), 
                    True, WHITE
                )
                surface.blit(value_text, (card_x + i * 25 + 5, card_y + 8))

            # Player total with soft/hard indicator
            total_text = f"Total: {player_value}"
            if is_soft:
                total_text += " (Soft)"
            elif self.contains_ace(player_hand):
                total_text += " (Hard)"
            
            total_render = self.font.render(total_text, True, self.color_palette['highlight'])
            surface.blit(total_render, (30, y_offset + 75))

            # --- Dealer Upcard ---
            dealer_value = self.get_card_value(dealer_up_card)
            
            # Dealer label
            dealer_label = self.font.render("Dealer Shows:", True, self.color_palette['text_primary'])
            surface.blit(dealer_label, (self.panel_width // 2 + 20, y_offset + 15))

            # Dealer card (miniature visual)
            pygame.draw.rect(surface, WHITE, (self.panel_width // 2 + 20, y_offset + 40, 20, 30), 1, border_radius=2)
            dealer_text = self.small_font.render(
                "A" if dealer_value == 1 else str(dealer_value), 
                True, WHITE
            )
            surface.blit(dealer_text, (self.panel_width // 2 + 25, y_offset + 48))

            # Dealer strength hint (weak/neutral/strong)
            strength = ""
            if dealer_value in [2, 3, 4, 5, 6]:
                strength = " (Weak)"
            elif dealer_value in [7, 8, 9]:
                strength = " (Neutral)"
            elif dealer_value in [10, 11]:
                strength = " (Strong)"
            
            dealer_strength = self.font.render(f"{dealer_value}{strength}", True, self.color_palette['highlight'])
            surface.blit(dealer_strength, (self.panel_width // 2 + 20, y_offset + 75))

        return section_height

    def _draw_recommendation_section(self, y_offset: int, player_hand: List[Tuple[str, str]], 
                                   dealer_up_card: Tuple[str, str]) -> int:
        """Draw recommendation with enhanced animations"""
        recommended = self.get_recommended_move(player_hand, dealer_up_card)
        action_key = recommended.split()[0][0] if recommended else 'H'  # First character

        ACTION_COLORS = {
            'H': HIT_COLOR,
            'S': STAND_COLOR,
            'D': DOUBLE_COLOR,
            'Y': SPLIT_COLOR,  # For split (Yes)
            'N': (100, 100, 100)  # For no split
        }

        rec_bg_color = ACTION_COLORS.get(action_key, (100, 100, 100))
        
        # Pulsing background
        pulse_intensity = 0.5 + 0.5 * math.sin(self.animation_state['pulse_phase'])
        pulse_color = (
            min(255, rec_bg_color[0] + int(50 * pulse_intensity)),
            min(255, rec_bg_color[1] + int(50 * pulse_intensity)),
            min(255, rec_bg_color[2] + int(50 * pulse_intensity))
        )
        
        # Draw section with rounded corners
        section_height = 60
        rect = pygame.Rect(20, y_offset, self.panel_width-40, section_height)
        pygame.draw.rect(self.surface, pulse_color, rect, border_radius=8)
        pygame.draw.rect(self.surface, self.color_palette['highlight'], rect, 2, border_radius=8)
        
        # Text with shadow for better visibility
        rec_text = self.title_font.render(f"Recommended: {recommended}", True, (0, 0, 0))
        shadow_text = self.title_font.render(f"Recommended: {recommended}", True, (100, 100, 100))
        
        text_x = self.panel_width//2 - rec_text.get_width()//2
        text_y = y_offset + section_height//2 - rec_text.get_height()//2
        
        self.surface.blit(shadow_text, (text_x+1, text_y+1))
        self.surface.blit(rec_text, (text_x, text_y))
        
        return section_height

    def _draw_probability_section(self, y_offset: int, player_hand: List[Tuple[str, str]], 
                                dealer_up_card: Tuple[str, str]) -> int:
        """Draw probability section with animated meters"""
        section_height = 120
        
        # Section background
        pygame.draw.rect(self.surface, (*self.color_palette['primary'], 150), 
                        (20, y_offset, self.panel_width-40, section_height))
        
        # Title
        title = self.font.render("Probability Analysis", True, self.color_palette['text_primary'])
        self.surface.blit(title, (30, y_offset + 5))
        
        # Draw EV meter
        self.draw_ev_meter(self.surface, 30, y_offset + 30, self.panel_width-60, 20)
        
        # Draw bust probability gauge
        gauge_x = self.panel_width - 80
        gauge_y = y_offset + 80
        self.draw_bust_prob_gauge(self.surface, gauge_x, gauge_y, 30)
        
        # Labels
        bust_label = self.font.render("Bust Chance:", True, self.color_palette['text_primary'])
        self.surface.blit(bust_label, (30, y_offset + 70))
        
        # Dealer bust probability if available
        if player_hand and dealer_up_card:
            risk = self.calculate_advanced_risk(player_hand, dealer_up_card)
            dealer_bust_text = f"Dealer Bust: {risk['dealer_bust_prob']*100:.1f}%"
            dealer_label = self.font.render(dealer_bust_text, True, self.color_palette['text_secondary'])
            self.surface.blit(dealer_label, (30, y_offset + 90))
        
        return section_height
    
    def _setup_logging(self):
        """Set up logging for strategy interactions and performance"""
        self.logger = logging.getLogger('StrategyAssistant')
        self.logger.setLevel(logging.INFO)
        
        # Create console and file handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('strategy_assistant.log')
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def set_visual_theme(self, theme_name: str):
        """
        Set the visual theme for the strategy assistant
        
        Args:
            theme_name (str): Name of the theme ('classic', 'dark_mode', 'high_contrast')
        """
        if theme_name not in self.visual_themes:
            self.logger.warning(f"Theme {theme_name} not found. Defaulting to classic.")
            theme_name = 'classic'
        
        self.current_theme = theme_name
        self.color_palette.update(self.visual_themes[theme_name])
        self.logger.info(f"Visual theme changed to: {theme_name}")

    def toggle_interaction_feature(self, feature: str, enabled: bool = None):
        """
        Toggle interaction features like hover effects or tooltips
        
        Args:
            feature (str): Feature to toggle ('hover_effects', 'tooltips')
            enabled (bool, optional): Force enable/disable. If None, toggle current state.
        """
        if feature not in self.interaction_state:
            raise ValueError(f"Unknown interaction feature: {feature}")
        
        if enabled is None:
            self.interaction_state[feature] = not self.interaction_state[feature]
        else:
            self.interaction_state[feature] = enabled
        
        status = "enabled" if self.interaction_state[feature] else "disabled"
        self.logger.info(f"{feature} {status}")

    def draw_decision_heatmap(self, surface, x, y, width, height):
        """
        Enhanced decision heatmap with more visual details
        
        Args:
            surface (pygame.Surface): Surface to draw on
            x (int): X coordinate
            y (int): Y coordinate
            width (int): Width of heatmap
            height (int): Height of heatmap
        """
        # Existing heatmap drawing logic with enhancements
        if len(self.decision_history) == 0:
            return
        
        # Draw background
        pygame.draw.rect(surface, self.color_palette['background'], (x, y, width, height))
        pygame.draw.rect(surface, self.color_palette['primary'], (x, y, width, height), 1)
        
        # Calculate blocks
        block_width = width // 10
        block_height = height // 5
        history = self.decision_history[-50:]  # Show last 50 decisions
        
        for i, decision in enumerate(history):
            if i >= 50:  # Only show 50 blocks
                break
            
            row = i // 10
            col = i % 10
            
            rect_x = x + col * block_width
            rect_y = y + row * block_height
            
            color = (0, 200, 0) if decision['correct'] else (200, 0, 0)
            pygame.draw.rect(surface, color, (rect_x, rect_y, block_width-1, block_height-1))
            
            # Add hover tooltips
            mouse_pos = pygame.mouse.get_pos()
            block_rect = pygame.Rect(rect_x, rect_y, block_width-1, block_height-1)
            if block_rect.collidepoint(mouse_pos):
                tooltip_text = (
                    f"Decision: {decision['recommended']} | "
                    f"Correct: {'Yes' if decision['correct'] else 'No'}"
                )
                self._render_advanced_tooltip(
                    surface, 
                    tooltip_text, 
                    (mouse_pos[0] + 10, mouse_pos[1] + 10)
                )
        
        # Add title
        title = self.font.render("Recent Decisions", True, self.color_palette['text_primary'])
        surface.blit(title, (x + width//2 - title.get_width()//2, y - 20))

    def draw_strategy_mode_selector(self, surface):
        """
        Interactive strategy mode selector with visual feedback
        
        Args:
            surface (pygame.Surface): Surface to draw on
        """
        modes = self.get_available_modes()
        mode_width = 100
        mode_height = 40
        spacing = 10
        
        total_width = len(modes) * (mode_width + spacing) - spacing
        start_x = (surface.get_width() - total_width) // 2
        
        mouse_pos = pygame.mouse.get_pos()
        
        for i, mode in enumerate(modes):
            x = start_x + i * (mode_width + spacing)
            y = surface.get_height() - 50
            
            mode_rect = pygame.Rect(x, y, mode_width, mode_height)
            
            # Highlight current mode and hover effects
            if mode == self.current_mode:
                pygame.draw.rect(surface, self.color_palette['highlight'], mode_rect)
            elif mode_rect.collidepoint(mouse_pos) and self.interaction_state['hover_effects']:
                pygame.draw.rect(surface, self.color_palette['primary'], mode_rect)
            else:
                pygame.draw.rect(surface, self.color_palette['background'], mode_rect)
            
            # Mode text
            font = pygame.font.Font(None, 20)
            text = font.render(mode.capitalize(), True, self.color_palette['text_primary'])
            text_rect = text.get_rect(center=mode_rect.center)
            surface.blit(text, text_rect)
            
            # Tooltips
            if mode_rect.collidepoint(mouse_pos):
                self._render_advanced_tooltip(
                    surface, 
                    self._tooltips['strategy_mode'], 
                    (x, y - 30)
                )

    def _initialize_base_strategies(self):
        """Initialize the base strategy tables"""
        self.hard_strategy = {
            21: ['S'] * 10,
            20: ['S'] * 10,
            19: ['S'] * 10,
            18: ['S'] * 10,
            17: ['S'] * 10,
            16: ['S'] * 7 + ['H'] * 3,
            15: ['S'] * 6 + ['H'] * 4,
            14: ['S'] * 6 + ['H'] * 4,
            13: ['S'] * 5 + ['H'] * 5,
            12: ['H'] * 3 + ['S'] * 4 + ['H'] * 3,
            11: ['D'] * 9 + ['H'],
            10: ['D'] * 8 + ['H'] * 2,
            9:  ['H'] * 2 + ['D'] * 5 + ['H'] * 3,
            8:  ['H'] * 10,
            7:  ['H'] * 10,
            6:  ['H'] * 10,
            5:  ['H'] * 10,
            4:  ['H'] * 10,
            17: ['S']*10 if not self.vip_mode else ['S']*6 + ['U']*4,  # Surrender 17 vs 9-A in VIP
            16: ['S']*7 + ['H']*3 if not self.vip_mode else ['U']*3 + ['S']*4 + ['H']*3,  # Surrender 16 vs 9-A
            15: ['S']*6 + ['H']*4 if not self.vip_mode else ['U']*1 + ['S']*5 + ['H']*4,  # Surrender 15 vs 10
        }

        self.deviations.update({
            (16, 9): (0, 'U'),  # Surrender 16 vs 9 at any count in VIP
            (16, 10): (0, 'U'),  # Surrender 16 vs 10 at any count in VIP
            (16, 1): (0, 'U'),   # Surrender 16 vs Ace at any count in VIP
            (15, 10): (0, 'U'),   # Surrender 15 vs 10 at any count in VIP
        })
        
        self.soft_strategy = {
            20: ['S'] * 10,
            19: ['S'] * 10,
            18: ['D'] * 6 + ['S'] * 2 + ['H'] * 2,
            17: ['H'] * 1 + ['D'] * 5 + ['H'] * 4,
            16: ['H'] * 1 + ['D'] * 5 + ['H'] * 4,
            15: ['H'] * 2 + ['D'] * 4 + ['H'] * 4,
            14: ['H'] * 3 + ['D'] * 3 + ['H'] * 4,
            13: ['H'] * 4 + ['D'] * 2 + ['H'] * 4
        }
        
        self.pair_strategy = {
            'A,A': ['Y'] * 10,
            '10,10': ['N'] * 10,
            '9,9': ['Y'] * 7 + ['N'] * 3,
            '8,8': ['Y'] * 10,
            '7,7': ['Y'] * 7 + ['N'] * 3,
            '6,6': ['Y'] * 6 + ['N'] * 4,
            '5,5': ['D'] * 8 + ['H'] * 2,
            '4,4': ['H'] * 8 + ['Y'] * 2,
            '3,3': ['Y'] * 7 + ['N'] * 3,
            '2,2': ['Y'] * 7 + ['N'] * 3
        }

    def _create_action_stats(self):
        """Create a standardized structure for tracking action statistics"""
        return {
            'total': 0,
            'correct': 0,
            'net_result': 0.0,
            'avg_result': 0.0
        }

    def _initialize_strategy_modes(self):
        """Create comprehensive strategy modes with refined configuration"""
        return {
            'conservative': {
                'risk_tolerance': 0.2,
                'description': "Cautious approach prioritizing bankroll preservation with reduced variance.",
                'hard_strategy_adjustments': {
                    12: ['S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-4, hit vs 5-A
                    13: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    14: ['S', 'S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    15: ['S', 'S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    16: ['S', 'S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                },
                'soft_strategy_adjustments': {
                    17: ['S'] * 10,  # Always stand on soft 17
                    18: ['S'] * 10,  # Always stand on soft 18
                    19: ['S'] * 10,  # Always stand on soft 19
                },
                'pair_strategy_adjustments': {
                    'A,A': ['Y'] * 10,  # Always split Aces
                    '8,8': ['Y'] * 10,  # Always split 8s
                    '9,9': ['S', 'Y', 'Y', 'Y', 'Y', 'S', 'S', 'S', 'S', 'S'],  # Split 9s vs 2-6
                    '7,7': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N'],  # Split 7s vs 2-7
                    '6,6': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 6s vs 2-6
                    '5,5': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # Never split 5s, double instead
                    '4,4': ['N', 'N', 'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N'],  # Split 4s only vs 5-6
                    '3,3': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 3s vs 2-6
                    '2,2': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 2s vs 2-6
                },
                'double_strategy_adjustments': {
                    9: ['N', 'D', 'D', 'D', 'D', 'D', 'N', 'N', 'N', 'N'],      # Double 9 vs 3-6
                    10: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'N', 'N'],     # Double 10 vs 2-8
                    11: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'N'],     # Double 11 vs 2-9
                    'A,2': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 13 vs 5-6
                    'A,3': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 14 vs 5-6
                    'A,4': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 15 vs 5-6
                    'A,5': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 16 vs 5-6
                    'A,6': ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 17 vs 4-6
                    'A,7': ['S', 'S', 'D', 'D', 'D', 'S', 'S', 'S', 'S', 'S'],  # Double soft 18 vs 4-6
                },
                'insurance': False,  # Never take insurance
                'surrender_strategy': {
                    16: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y'],  # Surrender 16 vs 9-10
                    15: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y'],  # Surrender 15 vs 10
                },
                'betting_strategy': {
                    'type': 'flat',  # Flat betting
                    'min_bet_multiplier': 1.0,
                    'max_bet_multiplier': 2.0,
                    'progression_rate': 0.1,  # Very slow progression
                    'negative_progression': False  # Don't increase bets after losses
                },
                    'count_thresholds': {
                    'high': 4,  # Only increase bets at very high counts
                    'low': -2,  # Decrease bets at moderately low counts
                }
            },

            'balanced': {
                'risk_tolerance': 0.5,
                'description': "Standard mathematically optimal strategy with moderate risk.",
                'hard_strategy_adjustments': {
                    12: ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 4-6, hit otherwise
                    13: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    14: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    15: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    16: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    17: ['S'] * 10,  # Always stand on 17+
                },
                'soft_strategy_adjustments': {
                    17: ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double vs 4-6, hit otherwise
                    18: ['S', 'S', 'S', 'S', 'D', 'S', 'S', 'S', 'S', 'S'],  # Double vs 6, stand otherwise
                    19: ['S'] * 10,  # Always stand on soft 19+
                },
                'pair_strategy_adjustments': {
                    'A,A': ['Y'] * 10,  # Always split Aces
                    '8,8': ['Y'] * 10,  # Always split 8s
                    '9,9': ['Y', 'Y', 'Y', 'Y', 'Y', 'S', 'Y', 'Y', 'S', 'S'],  # Split 9s vs all but 7,10,A
                    '7,7': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N'],  # Split 7s vs 2-7
                    '6,6': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 6s vs 2-6
                    '5,5': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # Never split 5s, double instead
                    '4,4': ['N', 'N', 'N', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N'],  # Split 4s only vs 5-6
                    '3,3': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 3s vs 2-6
                    '2,2': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 2s vs 2-6
                },
                'double_strategy_adjustments': {
                    9: ['D', 'D', 'D', 'D', 'D', 'D', 'N', 'N', 'N', 'N'],      # Double 9 vs 2-6
                    10: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'N', 'N'],     # Double 10 vs 2-8
                    11: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],     # Double 11 vs any
                    'A,2': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 13 vs 5-6
                    'A,3': ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 14 vs 5-6
                    'A,4': ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 15 vs 4-6
                    'A,5': ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 16 vs 4-6
                    'A,6': ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # Double soft 17 vs 3-6
                    'A,7': ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'S', 'S', 'S'],  # Double soft 18 vs 3-6
                },
                'insurance': False,  # Never take insurance
                'surrender_strategy': {
                    16: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y'],  # Surrender 16 vs 9-10
                    15: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y'],  # Surrender 15 vs 10
                },
                'betting_strategy': {
                    'type': 'spread',  # Spread betting based on count
                    'min_bet_multiplier': 1.0,
                    'max_bet_multiplier': 4.0,
                    'progression_rate': 0.3,  # Moderate progression
                    'negative_progression': False  # Don't increase bets after losses
                },
                'count_thresholds': {
                    'high': 2,  # Increase bets at +2 count
                    'low': -1,  # Decrease bets at -1 count
                }
            },

            'aggressive': {
                'risk_tolerance': 0.8,
                'description': "High-risk strategy maximizing potential returns with advanced counting techniques.",
                'hard_strategy_adjustments': {
                    12: ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 4-6, hit otherwise
                    13: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    14: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    15: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    16: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],  # Stand vs 2-6, hit vs 7-A
                    17: ['S'] * 10,  # Always stand on 17+
                },
                'soft_strategy_adjustments': {
                    17: ['H', 'D', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double vs 2-6, hit otherwise
                    18: ['S', 'S', 'D', 'D', 'D', 'D', 'S', 'S', 'S', 'S'],  # Double vs 3-6, stand otherwise
                    19: ['S'] * 10,  # Always stand on soft 19+
                },
                'pair_strategy_adjustments': {
                    'A,A': ['Y'] * 10,  # Always split Aces
                    '8,8': ['Y'] * 10,  # Always split 8s
                    '9,9': ['Y', 'Y', 'Y', 'Y', 'Y', 'S', 'Y', 'Y', 'S', 'S'],  # Split 9s vs all but 7,10,A
                    '7,7': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N'],  # Split 7s vs 2-7
                    '6,6': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],  # Split 6s vs 2-6
                    '5,5': ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N'],  # Never split 5s, double instead
                    '4,4': ['N', 'N', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N'],  # Split 4s vs 4-6
                    '3,3': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N'],  # Split 3s vs 2-7
                    '2,2': ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N'],  # Split 2s vs 2-7
                },
                'double_strategy_adjustments': {
                    8: ['N', 'N', 'N', 'N', 'D', 'D', 'N', 'N', 'N', 'N'],      # Double 8 vs 5-6
                    9: ['D', 'D', 'D', 'D', 'D', 'D', 'N', 'N', 'N', 'N'],      # Double 9 vs 2-6
                    10: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'N'],     # Double 10 vs 2-9
                    11: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],     # Double 11 vs any
                    'A,2': ['H', 'H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double soft 13 vs 4-6
                    'A,3': ['H', 'H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double soft 14 vs 4-6
                    'A,4': ['H', 'H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double soft 15 vs 4-6
                    'A,5': ['H', 'D', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double soft 16 vs 3-6
                    'A,6': ['D', 'D', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H'],  # Double soft 17 vs 2-6
                    'A,7': ['D', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H'],  # Double soft 18 vs 2-5, hit vs 8-10
                },
                'insurance': {
                    'threshold': 3,  # Take insurance when count is >= 3
                },
                'surrender_strategy': {
                    17: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y'],  # Surrender 17 vs A
                    16: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y', 'Y'],  # Surrender 16 vs 8-10
                    15: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y', 'Y'],  # Surrender 15 vs 9-10
                    14: ['N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'Y'],  # Surrender 14 vs 10
                },
                'betting_strategy': {
                    'type': 'aggressive_spread',  # Very large betting spread
                    'min_bet_multiplier': 1.0,
                    'max_bet_multiplier': 10.0,
                    'progression_rate': 0.5,  # Fast progression
                    'negative_progression': True  # Optional Martingale elements on strong counts
                },
                'count_thresholds': {
                    'high': 1,  # Increase bets at +1 count
                    'low': -2,  # Decrease bets below -2 count
                },
                'deviations': {  # Strategy deviations based on count
                    'index_plays': {
                        '16vs10': {
                            'default': 'H',
                            'count': 0,  # Stand on 16vs10 if count >= 0
                            'deviation': 'S'
                        },
                        '12vs3': {
                            'default': 'H',
                            'count': 2,  # Stand on 12vs3 if count >= 2
                            'deviation': 'S'
                        },
                        'insurance': {
                            'default': False,
                            'count': 3,  # Take insurance if count >= 3
                            'deviation': True
                        }
                    }
                }
            }
        }

    def _apply_mode_configuration(self, mode):
        """Apply the configuration for the specified mode"""
        current_mode_config = self.strategy_modes[mode]
        
        # Reset to base strategy first
        self._initialize_base_strategies()
        
        # Apply mode-specific adjustments
        if 'hard_strategy_adjustments' in current_mode_config:
            for hand, adjustment in current_mode_config['hard_strategy_adjustments'].items():
                self.hard_strategy[hand] = adjustment
        
        if 'soft_strategy_adjustments' in current_mode_config:
            for hand, adjustment in current_mode_config['soft_strategy_adjustments'].items():
                self.soft_strategy[hand] = adjustment
        
        if 'pair_strategy_adjustments' in current_mode_config:
            for pair, adjustment in current_mode_config['pair_strategy_adjustments'].items():
                self.pair_strategy[pair] = adjustment
        
        # Adjust betting strategy
        betting_multiplier = current_mode_config.get('betting_multiplier', 1.0)
        self.betting_strategy = [
            (threshold, int(units * betting_multiplier)) 
            for threshold, units in self.original_betting_strategy
        ]

    def set_vip_mode(self, enabled):
        """Enable or disable VIP mode which includes surrender recommendations"""
        self.vip_mode = enabled
        if enabled:
            # Adjust strategy tables for VIP mode
            self._initialize_base_strategies()
            logging.info("VIP mode activated - surrender recommendations enabled")
        else:
            # Revert to standard strategy
            self._initialize_base_strategies()
            logging.info("VIP mode deactivated")

    def calculate_advanced_risk(self, player_hand: List[Tuple[str, str]], dealer_card: Tuple[str, str]) -> Dict[str, float]:
        """
        Enhanced risk calculation with multiple factors
        
        Returns:
            Dict containing:
            - bust_probability: Chance of player busting if they hit
            - expected_value: Expected value of current hand
            - aggression_level: Recommended aggression level based on situation
            - dealer_bust_prob: Probability of dealer busting
        """
        hand_value = self.calculate_hand(player_hand)
        dealer_value = self.get_card_value(dealer_card)
        num_cards = len(player_hand)
        is_soft = self.contains_ace(player_hand) and hand_value <= 21
        
        # Comprehensive risk assessment
        risk_factors = {
            'hand_value': hand_value,
            'dealer_upcard': dealer_value,
            'hand_composition': num_cards,
            'contains_ace': is_soft,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration
        }
        
        # Advanced bust probability calculation
        bust_probability = self._calculate_bust_probability(risk_factors)
        
        # Expected value calculation
        expected_value = self._calculate_expected_value(risk_factors)
        
        # Dealer bust probability
        dealer_bust_prob = self.risk_assessment.calculate_dealer_bust_probability(dealer_value)
        
        # Determine aggression level
        if expected_value > 0.5:
            aggression_level = 'aggressive'
        elif expected_value > 0:
            aggression_level = 'balanced'
        else:
            aggression_level = 'conservative'
        
        return {
            'bust_probability': bust_probability,
            'expected_value': expected_value,
            'aggression_level': aggression_level,
            'dealer_bust_prob': dealer_bust_prob
        }
    
    def _calculate_bust_probability(self, risk_factors: Dict) -> float:
        """Calculate nuanced bust probability considering multiple factors"""
        hand_value = risk_factors['hand_value']
        contains_ace = risk_factors['contains_ace']
        num_cards = risk_factors['hand_composition']
        true_count = risk_factors['true_count']
        
        # Base bust probability
        if contains_ace:
            # Soft hand has lower bust probability
            base_prob = max(0, min(1, (hand_value - 11) / 10))
        else:
            # Hard hand bust probability
            base_prob = max(0, min(1, (hand_value - 11) / 10))
        
        # Adjust for number of cards (more cards = higher bust chance)
        card_adjustment = (num_cards - 2) * 0.05
        base_prob += card_adjustment
        
        # Adjust for true count (higher count = lower bust chance)
        count_adjustment = true_count * 0.05
        
        # Adjust for deck penetration (deeper penetration = more predictable)
        penetration_adjustment = risk_factors['deck_penetration'] * 0.1
        
        final_prob = base_prob - count_adjustment - penetration_adjustment
        return max(0, min(1, final_prob))
    
    def _calculate_expected_value(self, risk_factors: Dict) -> float:
        """Calculate more nuanced expected value considering multiple factors"""
        hand_value = risk_factors['hand_value']
        dealer_upcard = risk_factors['dealer_upcard']
        true_count = risk_factors['true_count']
        contains_ace = risk_factors['contains_ace']
        
        # Base expected value calculation
        if hand_value == 21:
            return 1.0  # Certain win for blackjack
        
        base_ev = 0.5  # Neutral starting point
        
        # Adjust based on true count
        count_multiplier = true_count * 0.1
        
        # Dealer upcard consideration
        dealer_weakness = 1 if dealer_upcard in [2, 3, 4, 5, 6] else 0.5
        
        # Hand value consideration
        if hand_value >= 17:
            hand_strength = 0.2
        elif hand_value >= 13:
            hand_strength = 0.1
        else:
            hand_strength = 0
        
        # Soft hand bonus
        soft_bonus = 0.1 if contains_ace else 0
        
        expected_value = (base_ev + count_multiplier + 
                         hand_strength * dealer_weakness + 
                         soft_bonus)
        
        return max(-1, min(1, expected_value))  # Clamp between -1 and 1

    def set_strategy_mode(self, mode: str):
        """Set the strategy mode with animation trigger"""
        available_modes = list(self.strategy_modes.keys())
        if mode not in available_modes:
            self.logger.warning(f"Invalid strategy mode '{mode}'. Available modes: {available_modes}")
            mode = available_modes[0]  # Fall back to first available mode

        if mode != self.current_mode:
            self.animation_state['mode_switch'] = 100  # Start animation
            self.logger.info(f"Strategy mode changing from {self.current_mode} to {mode}")
            
        self.current_mode = mode
        self._apply_mode_configuration(mode)
        
        # Update color palette with mode-specific accent color
        if 'color' in self.strategy_modes[mode]:
            self.color_palette['highlight'] = self.strategy_modes[mode]['color']

    def get_available_modes(self):
        """Return list of available strategy modes"""
        return list(self.strategy_modes.keys())
    
    def get_current_mode(self):
        """Return the current strategy mode"""
        return self.current_mode
    
    def get_mode_description(self, mode=None):
        """Get a description of the strategy mode"""
        if mode is None:
            mode = self.current_mode
        return self.strategy_modes.get(mode, {}).get('description', "Unknown strategy mode")

    def cycle_strategy_mode(self):
        """Cycle strategy modes in sequence"""
        modes = list(self.strategy_modes.keys())
        current_index = modes.index(self.current_mode)
        next_index = (current_index + 1) % len(modes)
        new_mode = modes[next_index]
        self.set_strategy_mode(new_mode)
        return new_mode

    def update_count(self, card: Tuple[str, str]):
        """Update the running count and true count based on the dealt card"""
        card_rank = card[0]
        
        # Update running count based on Hi-Lo system
        if card_rank in ['2', '3', '4', '5', '6']:
            self.running_count += 1
        elif card_rank in ['10', 'jack', 'queen', 'king', 'ace']:
            self.running_count -= 1
            
        # Update cards seen and deck penetration
        self.cards_seen += 1
        remaining_decks = (self.total_cards - self.cards_seen) / 52
        self.true_count = self.running_count / max(1, remaining_decks)
        
        # Update side counts for more advanced strategies
        if card_rank == 'ace':
            self.advanced_counting['ace_side_count'] += 1
        elif card_rank in ['10', 'jack', 'queen', 'king']:
            self.advanced_counting['ten_side_count'] += 1
            
        self.logger.debug(f"Card {card} dealt. Running count: {self.running_count}, True count: {self.true_count:.2f}")
        
    def reset_count(self):
        """Reset the count at the start of a new shoe"""
        self.running_count = 0
        self.true_count = 0
        self.cards_seen = 0
        self.deck_penetration = 0
        
    def get_bet_amount(self, min_bet: int, max_bet: int) -> int:
        """Calculate optimal bet based on true count and current strategy"""
        if self.true_count <= 1:
            return min_bet
            
        # Scale bet based on true count and strategy mode
        base_units = {
            'conservative': 1,
            'balanced': 2,
            'aggressive': 4
        }.get(self.current_mode, 1)
        
        # Adjust units based on true count
        units = base_units * min(10, max(1, int(abs(self.true_count))))
        
        # Calculate bet amount within min/max bounds
        bet = min(max_bet, max(min_bet, units * min_bet))
        
        # Apply risk tolerance from current strategy mode
        risk_factor = self.strategy_modes[self.current_mode]['risk_tolerance']
        final_bet = int(bet * risk_factor)
        
        self.logger.info(f"Recommended bet: {final_bet} (True count: {self.true_count:.2f}, Units: {units})")
        return final_bet
        
    def check_deviation(self, player_total: int, dealer_upcard: Tuple[str, str], basic_action: str) -> str:
        """Enhanced deviation check with composition-dependent strategy"""
        dealer_val = self.get_card_value(dealer_upcard)
        key = (player_total, dealer_val)
        
        # Check standard deviations first
        if key in self.deviations:
            threshold, deviation_action = self.deviations[key]
            if self.true_count >= threshold:
                return deviation_action
        
        # Check composition-dependent exceptions
        if (player_total == 16 and dealer_val == 10 and 
            self.advanced_counting['ten_side_count'] / self.cards_seen < 0.3):
            return 'S'  # More likely to stand if few tens remain
            
        if (player_total == 15 and dealer_val == 10 and 
            self.true_count > 3 and self.advanced_counting['ace_side_count'] > 2):
            return 'S'  # Stand with surplus aces
            
        return basic_action
        
    def check_pair_deviation(self, pair_key, dealer_upcard, basic_action):
        """Check deviations for pair splitting"""
        dealer_val = self.get_card_value(dealer_upcard)
        key = (pair_key, dealer_val)
        
        if key in self.deviations:
            threshold, deviation_action = self.deviations[key]
            if self.true_count >= threshold:
                return deviation_action
                
        return basic_action
    
    def draw_decision_history(self, surface: pygame.Surface, x: int, y: int, width: int) -> int:
        """Enhanced decision history visualization"""
        if not self.decision_history:
            return 0
            
        section_height = 150
        header_height = 30
        
        # Section background
        pygame.draw.rect(surface, (*self.color_palette['primary'], 150), 
                        (x, y, width, section_height))
        
        # Title with stats
        accuracy = self.get_detailed_strategy_accuracy()['RECENT']
        title_text = f"Decision History (Recent: {accuracy['accuracy_percentage']:.1f}%)"
        title = self.font.render(title_text, True, self.color_palette['text_primary'])
        surface.blit(title, (x + 10, y + 5))
        
        # Timeline visualization
        timeline_y = y + header_height + 10
        timeline_height = section_height - header_height - 20
        pygame.draw.rect(surface, (50, 50, 50), (x + 10, timeline_y, width - 20, timeline_height))
        
        # Plot decisions
        history = self.decision_history[-20:]  # Show last 20 decisions
        for i, decision in enumerate(history):
            x_pos = x + 10 + i * ((width - 20) / len(history))
            color = (0, 200, 0) if decision['correct'] else (200, 0, 0)
            height = timeline_height * (decision['result'] + 1) / 2  # Normalize result to 0-1
            
            pygame.draw.rect(surface, color, 
                           (x_pos, timeline_y + timeline_height - height, 
                            (width - 20) / len(history), height))
        
        return section_height
        
    def toggle(self):
        """Toggle visibility of the strategy assistant"""
        self.active = not self.active
        self.logger.info(f"Strategy assistant {'activated' if self.active else 'deactivated'}")

    def reset(self):
        """Reset all tracking statistics"""
        self.running_count = 0
        self.true_count = 0
        self.cards_seen = 0
        self.deck_penetration = 0
        self.decision_history = []
        
        for action in self.decision_stats:
            self.decision_stats[action] = self._create_action_stats()
        
        self.logger.info("Strategy assistant reset")
        
    def get_action_color(self, action):
        action_key = action[0].upper() if action else 'H'  # Default to HIT if no action
        action_map = {
            'S': STAND_COLOR,
            'H': HIT_COLOR,
            'D': DOUBLE_COLOR,
            'Y': (0, 200, 200),
            'N': (100, 100, 100),
            'U': SURRENDER_COLOR  # Add surrender color
        }
        return action_map.get(action_key, (100, 100, 100))
    
    def log_decision(self, recommended, player_decision, result=0):
        """Track decision statistics"""
        try:
            action_key = recommended.split()[0].upper()
            
            # Update decision statistics
            if action_key in self.decision_stats:
                stats = self.decision_stats[action_key]
                stats['total'] += 1
                stats['net_result'] += result
                stats['avg_result'] = stats['net_result'] / stats['total'] if stats['total'] > 0 else 0
                
                # Track decision correctness
                decision_correct = player_decision.upper() == action_key
                if decision_correct:
                    stats['correct'] += 1
            
            # Store decision history
            self.decision_history.append({
                'recommended': recommended,
                'player_decision': player_decision,
                'correct': decision_correct,
                'result': result,
                'timestamp': pygame.time.get_ticks()
            })
            
            # Limit history size
            if len(self.decision_history) > 200:
                self.decision_history.pop(0)
            
            self.last_decision_correct = decision_correct
        
        except Exception as e:
            print(f"Error logging decision: {e}")

    def get_detailed_strategy_accuracy(self):
        detailed_accuracy = {}
        for action, stats in self.decision_stats.items():
            if stats['total'] > 0:
                detailed_accuracy[action] = {
                    'total_decisions': stats['total'],
                    'correct_decisions': stats['correct'],
                    'accuracy_percentage': (stats['correct'] / stats['total']) * 100,
                    'average_result': stats['avg_result']
                }
            else:
                detailed_accuracy[action] = {
                    'total_decisions': 0,
                    'correct_decisions': 0,
                    'accuracy_percentage': 0,
                    'average_result': 0
                }
        
        # Add recent performance (last 20 decisions)
        recent_correct = sum(1 for d in self.decision_history[-20:] if d['correct'])
        recent_total = min(20, len(self.decision_history))
        detailed_accuracy['RECENT'] = {
            'accuracy_percentage': (recent_correct / recent_total * 100) if recent_total > 0 else 0,
            'sample_size': recent_total
        }
        
        return detailed_accuracy
    
    def get_recommended_move(self, player_hand, dealer_card):
        """Get recommended move based on basic strategy with deviations"""
        try:
            if not dealer_card:
                return "WAIT FOR DEALER CARD"
            
            if not player_hand:
                return "PLACE YOUR BET"
                
            dealer_val = self.get_card_value(dealer_card)
            dealer_index = dealer_val - 2 if dealer_val != 1 else 9

            # Check for surrender opportunity (only in VIP mode on first decision)
            if (self.vip_mode and len(player_hand) == 2 and 
                not any(hit for hit in self.decision_history[-5:] if hit['player_decision'] == 'HIT')):
                
                hand_value = self.calculate_hand(player_hand)
                
                # Check surrender strategy
                if hand_value in self.hard_strategy:
                    action_code = self.hard_strategy[hand_value][dealer_index]
                    if action_code == 'U':  # Surrender
                        return "SURRENDER"
                
                # Check surrender deviations
                key = (hand_value, dealer_val)
                if key in self.deviations:
                    threshold, deviation_action = self.deviations[key]
                    if deviation_action == 'U' and self.true_count >= threshold:
                        return "SURRENDER"

            # Hand value determination
            hand_value = self.calculate_hand(player_hand)
            is_soft = self.contains_ace(player_hand) and hand_value <= 21 and any(card[0] == 'ace' for card in player_hand if self.get_card_value(card) == 11)
            
            # Pair splitting logic
            if len(player_hand) == 2 and self.get_card_value(player_hand[0]) == self.get_card_value(player_hand[1]):
                card1_val = min(self.get_card_value(player_hand[0]), 10)
                card2_val = min(self.get_card_value(player_hand[1]), 10)
                pair_key = f"{card1_val},{card2_val}"
                
                if pair_key in self.pair_strategy:
                    basic_action = self.pair_strategy[pair_key][dealer_index]
                    # Check for deviations
                    action = self.check_pair_deviation(pair_key, dealer_card, basic_action)
                    
                    if action == 'Y':
                        return "SPLIT"
                    elif pair_key == '5,5' and action == 'D':
                        # Special case: treat two 5s as 10 (double rather than split)
                        hand_value = self.calculate_hand(player_hand)
                        if hand_value in self.hard_strategy:
                            action_code = self.hard_strategy[hand_value][dealer_index]
                            return "DOUBLE" if action_code == 'D' and len(player_hand) == 2 else "HIT"
                
            # Strategy selection
            strategy_table = self.soft_strategy if is_soft else self.hard_strategy
            
            if hand_value not in strategy_table:
                basic_recommendation = "HIT" if hand_value < 21 else "STAND"
            else:
                basic_action_code = strategy_table[hand_value][dealer_index]
                # Check for deviations
                action_code = self.check_deviation(hand_value, dealer_card, basic_action_code)
                
                action_map = {
                    'S': "STAND",
                    'H': "HIT",
                    'D': "DOUBLE" if len(player_hand) == 2 else "HIT"  # Only allow double on first move
                }
                basic_recommendation = action_map.get(action_code, "HIT")
            
            # Incorporate advanced risk assessment
            risk_analysis = self.calculate_advanced_risk(player_hand, dealer_card)
            
            # Annotate recommendation with confidence
            confidence_levels = {
                0.4: "Conservative",
                0.6: "Balanced",
                0.8: "Aggressive"
            }
            confidence = next(
                (conf_level for threshold, conf_level in 
                 sorted(confidence_levels.items()) if risk_analysis['expected_value'] <= threshold), 
                "Very Aggressive"
            )
            
            return f"{basic_recommendation} ({confidence})"
        
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return "ERROR"

    def update_visual_effects(self):
        # Pulse effect for recommendation
        self.pulse_alpha += 2 * self.pulse_direction
        if self.pulse_alpha >= 30 or self.pulse_alpha <= 0:
            self.pulse_direction *= -1
        
        # General animation counter
        self.animation_counter = (self.animation_counter + 1) % 360
        
        # Draw background
        pygame.draw.rect(surface, (30, 30, 30), (x, y, width, height))
        pygame.draw.rect(surface, (100, 100, 100), (x, y, width, height), 1)
        
        # Calculate blocks
        block_width = width // 10
        block_height = height // 5
        history = self.decision_history[-50:]  # Show last 50 decisions
        
        for i, decision in enumerate(history):
            if i >= 50:  # Only show 50 blocks
                break
            
            row = i // 10
            col = i % 10
            
            rect_x = x + col * block_width
            rect_y = y + row * block_height
            
            color = (0, 200, 0) if decision['correct'] else (200, 0, 0)
            pygame.draw.rect(surface, color, (rect_x, rect_y, block_width-1, block_height-1))
        
        # Add title
        title = self.font.render("Recent Decisions", True, (200, 200, 200))
        surface.blit(title, (x + width//2 - title.get_width()//2, y - 20))

    def _draw_probability_overlay(self, screen, player_hand, dealer_up_card):
        """Draw a probability overlay showing bust and success chances"""
        # Calculate probabilities
        risk_analysis = self.calculate_advanced_risk(player_hand, dealer_up_card)
        
        # Draw probability bars
        bar_width = 100
        bar_height = 10
        x_pos = STRATEGY_WIDTH - bar_width - 20
        y_pos = 50
        
        # Bust probability
        pygame.draw.rect(self.surface, self.color_palette['action_colors']['STAND'], 
                    (x_pos, y_pos, bar_width, bar_height))
        pygame.draw.rect(self.surface, self.color_palette['action_colors']['HIT'], 
                    (x_pos, y_pos, bar_width * (1 - risk_analysis['bust_probability']), bar_height))
        
        # Expected value
        y_pos += 20
        pygame.draw.rect(self.surface, self.color_palette['primary'], 
                    (x_pos, y_pos, bar_width, bar_height))
        pygame.draw.rect(self.surface, self.color_palette['highlight'], 
                    (x_pos, y_pos, bar_width * risk_analysis['expected_value'], bar_height))
        
        # Labels
        bust_label = self.font.render(f"Bust: {risk_analysis['bust_probability']*100:.1f}%", 
                                True, self.color_palette['text_primary'])
        ev_label = self.font.render(f"EV: {risk_analysis['expected_value']:.2f}", 
                              True, self.color_palette['text_primary'])
        
        self.surface.blit(bust_label, (x_pos - 80, y_pos))
        self.surface.blit(ev_label, (x_pos - 80, y_pos + 20))
        
        recommended = self.get_recommended_move(player_hand, dealer_up_card)
        rec_bg_color = self.get_action_color(recommended.split()[0] if recommended else 'H') or (100, 100, 100)  # Default to gray if None
            
        self.surface.fill((*self.color_palette['background'], 220))
        pygame.draw.rect(self.surface, self.color_palette['highlight'], (0, 0, STRATEGY_WIDTH, STRATEGY_HEIGHT), 2)
        
        # Update visual effects
        self.update_visual_effects()
        
        # Draw title with subtle animation
        title_color = self.color_palette['text_primary'] if self.animation_counter < 180 else \
                     self.color_palette['highlight']
        title = self.title_font.render("Blackjack Strategy Advisor", True, title_color)
        self.surface.blit(title, (STRATEGY_WIDTH//2 - title.get_width()//2, 10))
        
        # Get current recommendation
        if not dealer_up_card or not player_hand:
            recommended = "Waiting for cards..."
            explanation = "Make your initial bet to begin"
        else:
            recommended = self.get_recommended_move(player_hand, dealer_up_card)
            
            if player_decision:
                self.log_decision(recommended, player_decision)
            
            explanations = {
                "HIT": "Take another card to improve your hand",
                "STAND": "Stay with your current total to avoid busting",
                "DOUBLE": "Double your bet and take exactly one more card (best on 10-11)",
                "SPLIT": "Split your pair into two separate hands (always split Aces and 8s)",
                "WAIT": "Place your bet to begin the hand"
            }
            explanation = explanations.get(recommended.split()[0], "")
        
        # Draw hand information with more detail
        info_y = 40
        if player_hand:
            hand_value = self.calculate_hand(player_hand)
            is_soft = self.contains_ace(player_hand) and hand_value <= 21 and any(card[0] == 'ace' for card in player_hand if self.get_card_value(card) == 11)
            
            hand_text = f"Your Hand: {hand_value}"
            if is_soft:
                hand_text += " (Soft)"
            elif self.contains_ace(player_hand):
                hand_text += " (Hard)"
            
            value_text = self.title_font.render(hand_text, True, self.color_palette['text_primary'])
            self.surface.blit(value_text, (20, 50))
            
            # Show card composition
            if len(player_hand) <= 4:  # Only show if not too many cards
                cards_text = ", ".join(str(self.get_card_value(card)) for card in player_hand)
                cards_render = self.font.render(f"Cards: {cards_text}", True, self.color_palette['text_secondary'])
                self.surface.blit(cards_render, (20, 75))
        
        # Draw dealer information with more context
        if dealer_up_card:
            dealer_val = self.get_card_value(dealer_up_card)
            dealer_text = f"Dealer Shows: {'ACE' if dealer_val == 1 else dealer_val}"
            
            # Add basic dealer odds
            if dealer_val in [7, 8, 9, 10, 1]:
                odds_text = " (Strong)"
            elif dealer_val in [2, 3, 4, 5, 6]:
                odds_text = " (Weak)"
            else:
                odds_text = ""
                
            dealer_render = self.title_font.render(dealer_text + odds_text, True, (255, 255, 255))
            self.surface.blit(dealer_render, (20, 100 if len(player_hand) > 4 else 120))
        
        # Draw recommendation with enhanced visual effect
        rec_bg_color = self.get_action_color(recommended[0] if recommended else 'H')
        pulse_color = (
            min(255, rec_bg_color[0] + self.pulse_alpha),
            min(255, rec_bg_color[1] + self.pulse_alpha),
            min(255, rec_bg_color[2] + self.pulse_alpha)
        )
        
        pygame.draw.rect(self.surface, pulse_color, (20, 150, STRATEGY_WIDTH-40, 50))
        pygame.draw.rect(self.surface, (255, 255, 255), (20, 150, STRATEGY_WIDTH-40, 50), 2)
        
        rec_text = self.title_font.render(f"Recommended: {recommended}", True, (0, 0, 0))
        self.surface.blit(rec_text, (STRATEGY_WIDTH//2 - rec_text.get_width()//2, 165))
        
        # Draw probability overlay if enabled
        if self.visualization_options['show_probability_overlay'] and player_hand and dealer_up_card:
            self._draw_probability_overlay(self.surface, player_hand, dealer_up_card)
        
        # Draw explanation with improved word wrapping
        if explanation:
            words = explanation.split()
            line = ""
            y_pos = 210
            for word in words:
                test_line = line + word + " "
                text_width, _ = self.font.size(test_line)
                if text_width > STRATEGY_WIDTH - 40:
                    text = self.font.render(line, True, (220, 220, 220))
                    self.surface.blit(text, (20, y_pos))
                    y_pos += 20
                    line = word + " "
                else:
                    line = test_line
            if line:
                text = self.font.render(line, True, (220, 220, 220))
                self.surface.blit(text, (20, y_pos))
        
        # Draw decision feedback if available
        if player_decision and self.last_decision_correct is not None:
            feedback_y = y_pos + 30
            feedback_text = "Good decision!" if self.last_decision_correct else "Consider following strategy"
            feedback_color = (100, 255, 100) if self.last_decision_correct else (255, 100, 100)
            
            feedback = self.font.render(feedback_text, True, feedback_color)
            self.surface.blit(feedback, (20, feedback_y))
            
            # Draw detailed accuracy stats with performance metrics
            accuracy_data = self.get_detailed_strategy_accuracy()
            accuracy_y = feedback_y + 25
            
            # Recent performance
            recent_acc = accuracy_data['RECENT']
            acc_text = f"Recent Accuracy: {recent_acc['accuracy_percentage']:.1f}% ({recent_acc['sample_size']} hands)"
            acc_render = self.font.render(acc_text, True, (200, 200, 255))
            self.surface.blit(acc_render, (20, accuracy_y))
            accuracy_y += 20
            
            # Action-specific stats
            for action in ['HIT', 'STAND', 'DOUBLE', 'SPLIT']:
                stats = accuracy_data[action]
                if stats['total_decisions'] > 0:
                    acc_text = (f"{action}: {stats['accuracy_percentage']:.1f}% correct | "
                               f"Avg: {stats['average_result']:+.1f} chips")
                    acc_render = self.font.render(acc_text, True, self.get_action_color(action[0]))
                    self.surface.blit(acc_render, (20, accuracy_y))
                    accuracy_y += 20
            
            # Draw heatmap of recent decisions
            if len(self.decision_history) > 0:
                self.draw_decision_heatmap(self.surface, 20, accuracy_y + 10, STRATEGY_WIDTH-40, 60)
                accuracy_y += 80
        
        # Add legend with more information
        legend_y = STRATEGY_HEIGHT - 130
        legend_title = self.font.render("Action Legend (Expected Value):", True, (255, 255, 255))
        self.surface.blit(legend_title, (20, legend_y))
        
        legend_items = [
            ("HIT", HIT_COLOR, "+0.12"),
            ("STAND", STAND_COLOR, "+0.08"),
            ("DOUBLE", DOUBLE_COLOR, "+0.18"),
            ("SPLIT", SPLIT_COLOR, "+0.15")
        ]
        
        for i, (label, color, ev) in enumerate(legend_items):
            x_pos = 20 + (i % 2) * 150
            y_pos = legend_y + 20 + (i // 2) * 25
            
            pygame.draw.rect(self.surface, color, (x_pos, y_pos, 15, 15))
            pygame.draw.rect(self.surface, (255, 255, 255), (x_pos, y_pos, 15, 15), 1)
            
            text = self.font.render(f"{label} {ev}", True, (255, 255, 255))
            self.surface.blit(text, (x_pos + 20, y_pos))
        
        mode_desc = self.get_mode_description()
        words = mode_desc.split()
        line = ""
        y_pos = legend_y + 80
        for word in words:
            test_line = line + word + " "
            text_width, _ = self.font.size(test_line)
            if text_width > STRATEGY_WIDTH - 40:
                text = self.font.render(line, True, (200, 200, 200))
                self.surface.blit(text, (20, y_pos))
                y_pos += 15
                line = word + " "
            else:
                line = test_line
        if line:
            text = self.font.render(line, True, (200, 200, 200))
            self.surface.blit(text, (20, y_pos))
        
        # Draw toggle hint with animation
        hint_alpha = 120 + int(127 * math.sin(self.animation_counter * math.pi / 180))
        hint_lines = [
            "Press 'S' to toggle strategy advice",
            f"Current Mode: {self.current_mode.upper()} (Press 'M' to change)"
        ]
        for i, line in enumerate(hint_lines):
            hint_surface = self.font.render(line, True, (200, 200, 200))
            hint_surface.set_alpha(hint_alpha)
            self.surface.blit(hint_surface, (20, STRATEGY_HEIGHT - 40 + i * 20))

        # Draw to screen
        screen.blit(self.surface, (20, screen.get_height() - STRATEGY_HEIGHT - 20))
    
    def get_card_value(self, card: Tuple[str, str]) -> int:
        """Get numeric value of a card"""
        card_type = card[0]
        if card_type == 'ace':
            return 1
        elif card_type in ['king', 'queen', 'jack']:
            return 10
        else:
            return int(card_type)
    
    def calculate_hand(self, hand):
        value = 0
        aces = 0
        
        for card in hand:
            card_val = self.get_card_value(card)
            if card_val == 1:
                aces += 1
                value += 11
            else:
                value += card_val
        
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
                
        return value
    
    def contains_ace(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand contains an ace counted as 11"""
        return any(card[0] == 'ace' and self.get_card_value(card) == 11 for card in hand)

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
                'pos':
                position,
                'velocity': (math.cos(angle) * speed, math.sin(angle) * speed),
                'size':
                size,
                'color':
                color,
                'start_time':
                pygame.time.get_ticks(),
                'lifetime':
                lifetime * 1000
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
                particle['pos'][0] +
                particle['velocity'][0] * 0.016,  # 16ms frame time
                particle['pos'][1] + particle['velocity'][1] * 0.016)

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
            particle_surf = pygame.Surface((radius * 2, radius * 2),
                                           pygame.SRCALPHA)
            color_with_alpha = (*particle['color'], int(alpha))

            # Draw antialiased circle
            pygame.gfxdraw.filled_circle(particle_surf, radius, radius, radius,
                                  color_with_alpha)
            pygame.gfxdraw.aacircle(particle_surf, radius, radius, radius,
                             color_with_alpha)

            screen.blit(particle_surf, (pos[0] - radius, pos[1] - radius))


# Load card images
card_images = {}
suits = ['hearts', 'diamonds', 'clubs', 'spades']
values = [
    '2', '3', '4', '5', '6', '7', '8', '9', '10', 'jack', 'queen', 'king',
    'ace'
]

for suit in suits:
    for value in values:
        filename = f"{value}_of_{suit}.png"
        card_images[(value, suit)] = load_image(
            os.path.join("assets", "cards", filename),
            (CARD_WIDTH, CARD_HEIGHT))

# Load chip images
chip_images = {}
for value in CHIP_VALUES:
    chip_images[value] = load_image(
        os.path.join("assets", "chips", f"chip_{value}.png"), (60, 60))
    
placeholder_chip = load_image(os.path.join("assets", "chips", "chip_placeholder.png"), (60, 60))

# Card values
card_values = {
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '10': 10,
    'jack': 10,
    'queen': 10,
    'king': 10,
    'ace': 11
}


# Shuffle deck
def shuffle_deck(decks=8):
    logging.info("Shuffling the deck...")
    deck = [(v, s) for s in suits for v in values] * decks
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
def draw_glowing_button(screen,
                        rect,
                        text,
                        text_color,
                        button_color,
                        glow_color,
                        glow_size=10,
                        pulse=True):
    # Calculate pulse
    current_time = pygame.time.get_ticks() / 1000.0
    pulse_factor = 0.5 + math.sin(current_time * 5) * 0.5 if pulse else 1.0

    # Draw outer glow (multiple layers with decreasing alpha)
    for i in range(glow_size, 0, -1):
        alpha = int(180 * (i / glow_size) * pulse_factor)
        expanded_rect = pygame.Rect(rect.left - i, rect.top - i,
                                    rect.width + i * 2, rect.height + i * 2)
        glow_surf = pygame.Surface((expanded_rect.width, expanded_rect.height),
                                   pygame.SRCALPHA)
        # Ensure glow_color is a 3-tuple (RGB) before adding alpha
        if len(glow_color) == 3:
            glow_color_with_alpha = (*glow_color, alpha)
        else:
            glow_color_with_alpha = glow_color  # Assume it's already RGBA
        pygame.draw.rect(glow_surf,
                         glow_color_with_alpha,
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
        pygame.draw.line(screen, color, (rect.left, rect.top + i),
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

        # Draw chip (use placeholder if the chip value is invalid)
        chip_img = chip_images.get(value, placeholder_chip)
        screen.blit(chip_img, (x, y))

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
        text_effects.append(
            TextEffect(
                f"Achievement Unlocked: {ACHIEVEMENTS[achievement_key]['name']}!",
                (WIDTH // 2, HEIGHT // 2 + 100),
                GOLD,
                duration=3.0))

        # Also create a particle effect for visual flair
        particle_effects = ParticleSystem((WIDTH // 2, HEIGHT // 2 + 100),
                                          GOLD,
                                          count=50,
                                          duration=2.0)
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
def check_achievements(game_state, result, player_hand, dealer_hand,
                       player_money, current_bet, achievements_unlocked,
                       text_effects, particle_systems, stats):
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
            particles = unlock_achievement("first_win", achievements_unlocked,
                                           text_effects)
            if particles:
                particle_systems.append(particles)

        # Five wins achievement
        if not ACHIEVEMENTS["five_wins"]["unlocked"]:
            PROGRESS_TRACKERS["five_wins"] += 1
            if PROGRESS_TRACKERS["five_wins"] >= PROGRESS_REQUIREMENTS[
                    "five_wins"]:
                particles = unlock_achievement("five_wins",
                                               achievements_unlocked,
                                               text_effects)
                if particles:
                    particle_systems.append(particles)

        # Lucky streak achievement
        if stats["consecutive_wins"] >= 3 and not ACHIEVEMENTS["lucky_streak"][
                "unlocked"]:
            particles = unlock_achievement("lucky_streak",
                                           achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Blackjack achievement
        if calculate_hand(player_hand) == 21 and len(
                player_hand
        ) == 2 and not ACHIEVEMENTS["blackjack"]["unlocked"]:
            particles = unlock_achievement("blackjack", achievements_unlocked,
                                           text_effects)
            if particles:
                particle_systems.append(particles)

        # Lowballer achievement
        if calculate_hand(player_hand
                          ) <= 5 and not ACHIEVEMENTS["lowballer"]["unlocked"]:
            particles = unlock_achievement("lowballer", achievements_unlocked,
                                           text_effects)
            if particles:
                particle_systems.append(particles)

        # Ace master achievement
        if 'ace' in [card[0] for card in player_hand]:
            if not ACHIEVEMENTS["ace_master"]["unlocked"]:
                PROGRESS_TRACKERS["ace_master"] += 1
                if PROGRESS_TRACKERS["ace_master"] >= PROGRESS_REQUIREMENTS[
                        "ace_master"]:
                    particles = unlock_achievement("ace_master",
                                                   achievements_unlocked,
                                                   text_effects)
                    if particles:
                        particle_systems.append(particles)

        # Lucky number 21 achievement
        if not ACHIEVEMENTS["lucky_number_21"]["unlocked"]:
            PROGRESS_TRACKERS["lucky_number_21"] += 1
            if PROGRESS_TRACKERS["lucky_number_21"] >= PROGRESS_REQUIREMENTS[
                    "lucky_number_21"]:
                particles = unlock_achievement("lucky_number_21",
                                               achievements_unlocked,
                                               text_effects)
                if particles:
                    particle_systems.append(particles)

    elif result == "BUST!":
        stats["bust_count"] += 1
        stats["consecutive_busts"] += 1
        stats["consecutive_wins"] = 0
        stats["consecutive_losses"] += 1

        # Bust artist achievement
        if stats["consecutive_busts"] >= 5 and not ACHIEVEMENTS["bust_artist"][
                "unlocked"]:
            particles = unlock_achievement("bust_artist",
                                           achievements_unlocked, text_effects)
            if particles:
                particle_systems.append(particles)

        # Unlucky streak achievement
        if stats["consecutive_losses"] >= 3 and not ACHIEVEMENTS[
                "unlucky_streak"]["unlocked"]:
            particles = unlock_achievement("unlucky_streak",
                                           achievements_unlocked, text_effects)
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
            if PROGRESS_TRACKERS["dealers_nightmare"] >= PROGRESS_REQUIREMENTS[
                    "dealers_nightmare"]:
                particles = unlock_achievement("dealers_nightmare",
                                               achievements_unlocked,
                                               text_effects)
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
            if PROGRESS_TRACKERS["push_master"] >= PROGRESS_REQUIREMENTS[
                    "push_master"]:
                particles = unlock_achievement("push_master",
                                               achievements_unlocked,
                                               text_effects)
                if particles:
                    particle_systems.append(particles)

    # Check for bet-related achievements
    if current_bet >= 500 and not ACHIEVEMENTS["high_roller"]["unlocked"]:
        particles = unlock_achievement("high_roller", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    if current_bet >= 500:
        if not ACHIEVEMENTS["big_spender"]["unlocked"]:
            PROGRESS_TRACKERS["big_spender"] += 1
            if PROGRESS_TRACKERS["big_spender"] >= PROGRESS_REQUIREMENTS[
                    "big_spender"]:
                particles = unlock_achievement("big_spender",
                                               achievements_unlocked,
                                               text_effects)
                if particles:
                    particle_systems.append(particles)

    if current_bet <= 10:
        if not ACHIEVEMENTS["small_bettor"]["unlocked"]:
            PROGRESS_TRACKERS["small_bettor"] += 1
            if PROGRESS_TRACKERS["small_bettor"] >= PROGRESS_REQUIREMENTS[
                    "small_bettor"]:
                particles = unlock_achievement("small_bettor",
                                               achievements_unlocked,
                                               text_effects)
                if particles:
                    particle_systems.append(particles)

    # Check for special card combinations
    if len(player_hand) == 2 and all(
            card[0] == 'ace' for card in
            player_hand) and not ACHIEVEMENTS["perfect_pair"]["unlocked"]:
        particles = unlock_achievement("perfect_pair", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    if len(player_hand) == 3 and all(
            card[0] == '7' for card in
            player_hand) and not ACHIEVEMENTS["lucky_7s"]["unlocked"]:
        particles = unlock_achievement("lucky_7s", achievements_unlocked,
                                       text_effects)
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

    if king_suit and queen_suit and king_suit == queen_suit and not ACHIEVEMENTS[
            "royal_flush"]["unlocked"]:
        particles = unlock_achievement("royal_flush", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    # Money-related achievements
    if player_money >= 1000000 and not ACHIEVEMENTS["millionaire"]["unlocked"]:
        particles = unlock_achievement("millionaire", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    if stats["total_winnings"] >= 10000 and not ACHIEVEMENTS["chip_collector"][
            "unlocked"]:
        particles = unlock_achievement("chip_collector", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    if player_money <= 10 and result == "YOU WIN!" and not ACHIEVEMENTS[
            "comeback_king"]["unlocked"]:
        particles = unlock_achievement("comeback_king", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)


# All-in achievement
    if current_bet == player_money and not ACHIEVEMENTS["all_in"]["unlocked"]:
        particles = unlock_achievement("all_in", achievements_unlocked,
                                       text_effects)
        if particles:
            particle_systems.append(particles)

    # Check for all achievements unlocked
    all_unlocked = True
    for key, achievement in ACHIEVEMENTS.items():
        if key != "blackjack_legend" and not achievement["unlocked"]:
            all_unlocked = False
            break

    if all_unlocked and not ACHIEVEMENTS["blackjack_legend"]["unlocked"]:
        particles = unlock_achievement("blackjack_legend",
                                       achievements_unlocked, text_effects)
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
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 50))

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
                pygame.draw.rect(screen, (50, 40, 0),
                                 box_rect,
                                 border_radius=8)
                pygame.draw.rect(screen, GOLD, box_rect, 2, border_radius=8)
                color = WHITE
            else:
                # Dark background for locked achievements
                pygame.draw.rect(screen, (40, 40, 40),
                                 box_rect,
                                 border_radius=8)
                pygame.draw.rect(screen, (100, 100, 100),
                                 box_rect,
                                 2,
                                 border_radius=8)
                color = (150, 150, 150)

            # Draw achievement name
            name_font = pygame.font.Font(None, 28)
            name_text = name_font.render(ACHIEVEMENTS[key]["name"], True,
                                         color)
            screen.blit(
                name_text,
                (x + (cell_width - 40 - name_text.get_width()) // 2, y + 15))

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
                screen.blit(line_surf,
                            (x +
                             (cell_width - 40 - line_surf.get_width()) // 2,
                             y + 50 + j * 25))

        # Draw back button
        back_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 100, 200, 50)
        draw_glowing_button(screen, back_button, "Back to Game", WHITE, BLUE,
                            (100, 100, 255))

        # Handle back button
        if pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if back_button.collidepoint(mouse_pos):
                running = False

        pygame.display.flip()
        clock.tick(60)


# Define the main_menu() function before it is called
def main_menu():
    global JACKPOT_ENABLED, JACKPOT_AMOUNT, PLAYER_MONEY, VIP_ROOM_ACTIVE

    PLAYER_MONEY = load_player_money()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack Deluxe - Menu")
    clock = pygame.time.Clock()

    # Initialize menu state
    menu_option = 0  # 0 = Play, 1 = Achievements, 2 = Jackpot, 3 = VIP Room, 4 = Quit

    # Menu loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    menu_option = (menu_option - 1) % 5  # 5 options now
                elif event.key == pygame.K_DOWN:
                    menu_option = (menu_option + 1) % 5  # 5 options now
                elif event.key == pygame.K_RETURN:
                    if menu_option == 0:
                        return "PLAY"
                    elif menu_option == 1:
                        return "ACHIEVEMENTS"
                    elif menu_option == 2:
                        # Toggle jackpot
                        JACKPOT_ENABLED = not JACKPOT_ENABLED
                        if not JACKPOT_ENABLED:
                            JACKPOT_AMOUNT = 0  # Reset jackpot if disabled
                    elif menu_option == 3:
                        # Enter VIP room if player has enough money
                        if PLAYER_MONEY >= 10000:  # VIP room minimum bet
                            VIP_ROOM_ACTIVE = True
                            return "PLAY"  # Start game in VIP mode
                        else:
                            # Show message if player doesn't have enough money
                            print("Not enough money to enter VIP room!")
                            # Display a message on the screen
                            message_font = pygame.font.Font(None, 36)
                            message_text = message_font.render("Not enough money to enter VIP room!", True, RED)
                            screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, HEIGHT // 2 + 200))
                            pygame.display.flip()
                            pygame.time.wait(2000)  # Show the message for 2 seconds
                    else:
                        VIP_ROOM_ACTIVE = False
                        return "QUIT"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()

                # Check if clicked on menu options
                play_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 50, 200, 50)
                achievements_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 20, 200, 50)
                jackpot_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 90, 200, 50)
                vip_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 160, 200, 50)
                quit_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 + 230, 200, 50)

                if play_button.collidepoint(mouse_pos):
                    return "PLAY"
                elif achievements_button.collidepoint(mouse_pos):
                    return "ACHIEVEMENTS"
                elif jackpot_button.collidepoint(mouse_pos):
                    # Toggle jackpot
                    JACKPOT_ENABLED = not JACKPOT_ENABLED
                    if not JACKPOT_ENABLED:
                        JACKPOT_AMOUNT = 0  # Reset jackpot if disabled
                elif vip_button.collidepoint(mouse_pos):
                    # Enter VIP room if player has enough money
                    if PLAYER_MONEY >= 10000:  # VIP room minimum bet
                        VIP_ROOM_ACTIVE = True
                        return "PLAY"  # Start game in VIP mode
                    else:
                        # Show message if player doesn't have enough money
                        print("Not enough money to enter VIP room!")
                        # Display a message on the screen
                        message_font = pygame.font.Font(None, 36)
                        message_text = message_font.render("Not enough money to enter VIP room!", True, RED)
                        screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, HEIGHT // 2 + 200))
                        pygame.display.flip()
                        pygame.time.wait(2000)  # Show the message for 2 seconds
                elif quit_button.collidepoint(mouse_pos):
                    VIP_ROOM_ACTIVE = False
                    return "QUIT"

        # Draw menu
        screen.fill(BLACK)

        # Draw animated background
        current_time = pygame.time.get_ticks() / 1000.0
        for i in range(50):
            x = (WIDTH // 2) + math.cos(current_time * 0.5 + i * 0.2) * (WIDTH // 3)
            y = (HEIGHT // 2) + math.sin(current_time * 0.5 + i * 0.2) * (HEIGHT // 3)
            size = 2 + math.sin(current_time + i) * 2
            color = (int(127 + 127 * math.sin(current_time * 0.7 + i * 0.1)),
                     int(127 + 127 * math.sin(current_time * 0.5 + i * 0.1)),
                     int(127 + 127 * math.sin(current_time * 0.3 + i * 0.1)))
            pygame.draw.circle(screen, color, (int(x), int(y)), int(size))

        # Draw title
        title_font = pygame.font.Font(None, 92)
        title_text = title_font.render("BLACKJACK DELUXE", True, GOLD)
        screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, HEIGHT // 4))

        # Draw player's money
        money_font = pygame.font.Font(None, 36)
        money_text = money_font.render(f"Money: ${PLAYER_MONEY}", True, GOLD)
        screen.blit(money_text, (WIDTH // 2 - money_text.get_width() // 2, HEIGHT // 4 + 60))

        # Draw menu options
        options = ["Play Game", "Achievements", f"Jackpot: {'ON' if JACKPOT_ENABLED else 'OFF'}", "VIP Room", "Quit"]
        for i, option in enumerate(options):
            color = GOLD if i == menu_option else WHITE
            button_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 50 + i * 70, 200, 50)

            if i == menu_option:
                draw_glowing_button(screen, button_rect, option, BLACK, color, (*color, 128), pulse=True)
            else:
                draw_glowing_button(screen, button_rect, option, BLACK, color, (*color, 64), pulse=False)

        # Update display
        pygame.display.flip()
        clock.tick(60)

def update_card_count(self, card):
    """Update the running count for card counting (Expert level)"""
    if self.difficulty != "Expert":
        return
        
    value = get_card_value(card)
    if value >= 10 or value == 1:  # High cards (10, J, Q, K, A)
        self.card_count -= 1
    elif 2 <= value <= 6:  # Low cards
        self.card_count += 1

# Add VIP Rooms and Currency Exchange classes
class VIPRoom:
    def __init__(self, min_bet=10000, max_bet=50000, special_rules=None):
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.special_rules = special_rules or {}
        self.active = False
        self.special_rules = {
            "double_after_split": True,
            "surrender": True,
            "resplit_aces": False,
            "perfect_pairs": True,
            "twenty_one_three": True
        }
        self.active = False
        self.deck_count = 4
        self.dealers = {
            "James": {"gender": "male", "greetings": []},
            "Sophia": {"gender": "female", "greetings": []},
            "Michael": {"gender": "male", "greetings": []},
            "Emma": {"gender": "female", "greetings": []},
            "William": {"gender": "male", "greetings": []}
        }
        self.current_greeting = None 
        self.current_dealer = None
        self.betting_presets = [100, 500, 1000, 5000, 10000]
        self.auto_play = False
        self.auto_play_delay = 2
        self.last_auto_play_time = 0

    def get_vip_deck(self):
        return shuffle_deck(decks=self.deck_count)

    def load_dealer_audio(self):
        """Load all dealer greeting audio files with error handling"""
        for dealer_name, dealer_data in self.dealers.items():
            dealer_data["greetings"] = []
        for dealer_name, dealer_data in self.dealers.items():
            try:
                dealer_folder = f"assets/sounds/dealers/{dealer_name.lower()}/"
                
                # Check if directory exists
                if not os.path.exists(dealer_folder):
                    print(f"Warning: No audio folder found for dealer {dealer_name}")
                    continue
                
                # Load all greeting files
                greetings = []
                for file in os.listdir(dealer_folder):
                    if file.startswith("greeting_") and (file.endswith(".wav") or file.endswith(".mp3")):
                        try:
                            sound = pygame.mixer.Sound(os.path.join(dealer_folder, file))
                            greetings.append(sound)
                            print(f"Loaded {file} for {dealer_name}")  # Debug
                        except Exception as e:
                            print(f"Error loading {file} for {dealer_name}: {e}")
                
                dealer_data["greetings"] = greetings
                if not greetings:
                    print(f"Warning: No valid audio files found for {dealer_name}")
                    
            except Exception as e:
                print(f"Error loading audio for {dealer_name}: {e}")
        
        # Set initial dealer
        self.change_dealer()

    def change_dealer(self):
        """Switch to a different dealer randomly and select a random greeting"""
        available_dealers = [d for d in self.dealers if self.dealers[d]["greetings"]]
        if not available_dealers:
            self.current_dealer = None
            self.current_greeting = None
            return None
            
        self.current_dealer = random.choice(available_dealers)
        self.current_greeting = random.choice(self.dealers[self.current_dealer]["greetings"])
        return self.current_greeting

    def get_random_greeting(self):
        """Select one random dealer and greeting to play"""
        # Get all dealers that have greetings loaded
        available_dealers = [d for d in self.dealers if self.dealers[d]["greetings"]]
        if not available_dealers:
            return None
            
        # Select random dealer
        self.current_dealer = random.choice(available_dealers)
        # Select random greeting from that dealer
        self.current_greeting = random.choice(self.dealers[self.current_dealer]["greetings"])
        return self.current_greeting

    def enter(self, player_money):
        if player_money < self.min_bet:
            return False, "Not enough money to enter VIP room."
        self.active = True
        return True, "Welcome to the VIP room!"

    def exit(self):
        self.active = False
        return "Exited VIP room."

    def apply_special_rules(self, game_state):
        if "double_after_split" in self.special_rules:
            # Allow doubling after splitting
            pass  # Add logic to handle this rule

# Create an instance of VIPRoom with default values
vip_room = VIPRoom()

# Or with custom values
# vip_room = VIPRoom(min_bet=20000, max_bet=100000)


class CurrencyExchange:
    def __init__(self):
        self.exchange_rate = 1.0  # 1 chip = 1 virtual currency unit
        self.collectibles = {}

    def convert_to_chips(self, amount):
        return int(amount * self.exchange_rate)

    def convert_to_collectibles(self, amount, collectible_type):
        if collectible_type in self.collectibles:
            self.collectibles[collectible_type] += amount
        else:
            self.collectibles[collectible_type] = amount

def game_over_screen():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Game Over")
    clock = pygame.time.Clock()

    # Load game over sound
    game_over_sound = pygame.mixer.Sound("assets/sounds/game_over.wav")
    game_over_sound.play()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Restart the game
                    return "RESTART"
                elif event.key == pygame.K_q:  # Quit the game
                    return "QUIT"
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if restart_button.collidepoint(mouse_pos):
                    return "RESTART"
                elif quit_button.collidepoint(mouse_pos):
                    return "QUIT"
            
        # Draw background
        screen.fill(BLACK)

        # Draw game over text
        game_over_font = pygame.font.Font(None, 92)
        game_over_text = game_over_font.render("GAME OVER", True, RED)
        screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 3))

        # Draw message
        message_font = pygame.font.Font(None, 36)
        message_text = message_font.render("You've run out of money!", True, WHITE)
        screen.blit(message_text, (WIDTH // 2 - message_text.get_width() // 2, HEIGHT // 2))

        # Draw restart and quit buttons
        restart_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 50)
        quit_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 170, 300, 50)

        draw_glowing_button(screen, restart_button, "Restart Game", WHITE, GREEN, (100, 255, 100))
        draw_glowing_button(screen, quit_button, "Quit", WHITE, RED, (255, 100, 100))

        # Update display
        pygame.display.flip()
        clock.tick(60)

# Main game function
def main():
    global perfect_pairs_bet, twenty_one_three_bet, PLAYER_MONEY, VIP_ROOM_ACTIVE
    PLAYER_MONEY = load_player_money()  # Load player money at the start

    strategy_assistant = StrategyAssistant()

    logging.info("Starting new game session")
    
    dealer_welcome = pygame.mixer.Sound("assets/sounds/dealer_welcome.wav")
    
    # Set up display
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackjack Deluxe")
    clock = pygame.time.Clock()
    perfect_pairs_button = pygame.Rect(50, HEIGHT - 240, 100, 40)
    twenty_one_three_button = pygame.Rect(WIDTH//2 + 100, HEIGHT - 240, 100, 40)

    # Initialize game state
    deck = shuffle_deck()
    player_hand = []
    dealer_hand = []
    text_effects = []
    current_bet = 0
    game_state = "BETTING"  # game_state options : "BETTING", "DEALING", "INSURANCE", "PLAYER_TURN", "DEALER_TURN", "GAME_OVER"
    player_hands = [[]]  # List of hands (for splitting)
    current_hand_index = 0  # Index of the hand player is currently playing
    insurance_bet = 0  # Insurance bet amount
    split_bets = [] 

    if VIP_ROOM_ACTIVE:
        vip_room = VIPRoom()
        deck = vip_room.get_vip_deck()
        vip_room.load_dealer_audio()
        print(f"VIP Room initialized with {len(vip_room.dealers)} dealers")  # Debug
        MIN_BET = vip_room.min_bet
        MAX_BET = min(vip_room.max_bet, PLAYER_MONEY)
        MIN_BET = 10000  # Set higher minimum bet for VIP room
        strategy_assistant.set_vip_mode(True)

        if not hasattr(main, 'vip_greeting_played'):  # Only play once per session
            try:
                greeting = vip_room.get_random_greeting()
                if greeting:
                    channel = pygame.mixer.find_channel()
                    if channel:
                        channel.play(greeting)
                        print(f"Playing greeting from {vip_room.current_dealer}")  # Debug
                        text_effects.append(TextEffect(
                                f"Dealer {vip_room.current_dealer} greets you",
                            (WIDTH//2, HEIGHT//4), 
                            WHITE
                        ))
                    else:
                        print("No available audio channels!")
                else:
                    print("No greeting sound available!")
            except Exception as e:
                print(f"Error playing VIP greeting: {e}")
            main.vip_greeting_played = True

        # Random greetings during gameplay
        if random.random() < 0.01:  # 1% chance per frame
            try:
                pygame.mixer.stop()
                greeting = vip_room.get_random_greeting()
                if greeting:
                    channel = pygame.mixer.find_channel()
                    if channel:
                        channel.play(greeting)
                        print(f"Playing random greeting from {vip_room.current_dealer}")
            except Exception as e:
                print(f"Error playing random greeting: {e}")
    else:
        MIN_BET = 10  # Regular table minimum bet
        MAX_BET = PLAYER_MONEY  # Regular table maximum bet
        strategy_assistant.set_vip_mode(False)
        dealer_welcome.play()

    if VIP_ROOM_ACTIVE and random.random() < 0.5:
        vip_room.get_random_greeting
        text_effects.append(TextEffect(
            f"Dealer {vip_room.current_dealer} greets you",
            (WIDTH//2, HEIGHT//4), 
            WHITE
        ))

    # Progressive Jackpot Variables
    JACKPOT_AMOUNT = 0  # Current jackpot amount
    JACKPOT_CONTRIBUTION = 0.01  # 1% of each bet contributes to the jackpot
    JACKPOT_TRIGGER = "natural_blackjack"  # Condition to win the jackpot

    # VIP Rooms
    vip_room = VIPRoom(min_bet=10000, max_bet=50000, special_rules={"double_after_split": True})

    # Currency Exchange
    currency_exchange = CurrencyExchange()

    # Animation tracking
    card_animations = []
    chip_animations = []
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
    hit_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT - 100, 100, 50)
    stand_button = pygame.Rect(WIDTH // 2, HEIGHT - 100, 100, 50)
    double_button = pygame.Rect(WIDTH // 2 + 150, HEIGHT - 100, 100, 50)
    deal_button = pygame.Rect(WIDTH // 2 + 100, HEIGHT - 100, 100, 50)
    split_button = pygame.Rect(WIDTH // 2 + 250, HEIGHT - 100, 100, 50)
    insurance_button = pygame.Rect(WIDTH // 2 - 200, HEIGHT - 100, 150, 50)
    no_insurance_button = pygame.Rect(WIDTH // 2 + 50, HEIGHT - 100, 150, 50)
    all_in_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 200, 200, 50)
    surrender_button = pygame.Rect(WIDTH // 2 + 450, HEIGHT - 100, 100, 50)

    # Results
    result = ""
    show_dealer_cards = False
    split_results = []  # Store results for each split hand

    def check_perfect_pairs(player_hand):
        if len(player_hand) != 2:
            return None
            
        card1, card2 = player_hand
        rank1, suit1 = card1
        rank2, suit2 = card2
        
        if rank1 != rank2:
            return None
            
        # Check perfect pair (same suit and rank)
        if suit1 == suit2:
            return "perfect"
        
        # Check colored pair (same color)
        suit_colors = {
            "hearts": "red",
            "diamonds": "red",
            "clubs": "black",
            "spades": "black"
        }
        if suit_colors[suit1] == suit_colors[suit2]:
            return "colored"
        
        # Mixed pair
        return "mixed"

    def check_twenty_one_three(player_hand, dealer_up_card):
        # Combine cards and validate
        cards = []
        try:
            cards.extend(player_hand[:2])  # First two player cards
            cards.append(dealer_up_card)   # Dealer's up card
        except (TypeError, IndexError):
            return "none"

        # Validate all cards have proper format
        for card in cards:
            if not isinstance(card, (tuple, list)) or len(card) < 2:
                return "none"

        # Get ranks and suits
        ranks = []
        suits = []
        for card in cards:
            try:
                ranks.append(card[0])
                suits.append(card[1])
            except (IndexError, TypeError):
                return "none"

        for card in cards:
            if not isinstance(card, (tuple, list)) or len(card) < 2:
                return "none"

        # Check suited triple (all same suit)
        if len(set(suits)) == 1:
            return "suited_triple"
        
        # Check three of a kind
        if len(set(ranks)) == 1:
            return "three_of_a_kind"
        
        # Convert ranks to numerical values for straight check
        rank_values = []
        for rank in ranks:
            if rank == 'ace':
                rank_values.append(1)
            elif rank in ['king', 'queen', 'jack', '10']:
                rank_values.append(10)
            else:
                try:
                    rank_values.append(int(rank))
                except ValueError:
                    return "none"  # Invalid card rank
        
        rank_values.sort()
    
        # Check straight flush (all same suit and sequential)
        if len(set(suits)) == 1:
            # Check for A-2-3 straight
            if set(rank_values) == {1, 2, 3}:
                return "straight_flush"
            # Check regular straight
            if (rank_values[1] == rank_values[0] + 1 and 
                rank_values[2] == rank_values[1] + 1):
                return "straight_flush"

        # Check straight
        if set(rank_values) == {1, 2, 3}:
            return "straight"
        if (rank_values[1] == rank_values[0] + 1 and 
            rank_values[2] == rank_values[1] + 1):
            return "straight"

        # Check flush
        if len(set(suits)) == 1:
            return "flush"

        return "none"
    
    def draw_tooltip(self, surface, text, button_rect):
        """Draw a tooltip above the specified button"""
        tooltip_font = pygame.font.Font(None, 24)
        text_surface = tooltip_font.render(text, True, WHITE)

        # Calculate tooltip position
        tooltip_rect = pygame.Rect(
            button_rect.centerx - text_surface.get_width() // 2 - 5,
            button_rect.y - 30,
            text_surface.get_width() + 10,
            25
        )

        # Draw background and border
        pygame.draw.rect(surface, (0, 0, 0, 200), tooltip_rect, border_radius=3)
        pygame.draw.rect(surface, (100, 100, 100), tooltip_rect, 1, border_radius=3)

        # Draw text
        surface.blit(text_surface, (tooltip_rect.x + 5, tooltip_rect.y + 5))

    # Game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:  # M key to change mode
                    new_mode = strategy_assistant.cycle_strategy_mode()
                    print(f"Strategy mode changed to: {new_mode}")
                elif event.key == pygame.K_s:  # S key to toggle
                    strategy_assistant.toggle()
                elif event.key == pygame.K_t:  # Press 'T' to cycle themes
                    current_theme_index = list(strategy_assistant.visual_themes.keys()).index(
                        strategy_assistant.current_theme)
                    next_theme = list(strategy_assistant.visual_themes.keys())[
                        (current_theme_index + 1) % len(strategy_assistant.visual_themes)]
                    strategy_assistant.set_visual_theme(next_theme)    

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
                                if current_bet + value <= MAX_BET and current_bet + value <= PLAYER_MONEY:
                                    current_bet += value
                                    logging.info(f"Player placed a bet of ${value}. Total bet: ${current_bet}")
                                    # Create chip animation
                                    chip_animations.append(
                                        ChipAnimation(
                                            value, (x, HEIGHT - 150),
                                            (WIDTH // 2, HEIGHT - 200)))
                                    chip_place.play()

                                    # Add to jackpot if enabled
                                    if JACKPOT_ENABLED:
                                        jackpot_contribution = int(current_bet * JACKPOT_CONTRIBUTION)
                                        JACKPOT_AMOUNT += jackpot_contribution
                                        PLAYER_MONEY -= jackpot_contribution
                                        save_player_money(PLAYER_MONEY)

                    # Check if player clicked on deal button
                    if deal_button.collidepoint(
                            mouse_pos) and current_bet >= MIN_BET:
                        game_state = "DEALING"

                        dealer_hit_or_stand = pygame.mixer.Sound(
                            "assets/sounds/hit_stand_double.wav")
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
                        card_animations.append(
                            create_deal_animation(player_card, deck_pos,
                                                  (WIDTH // 2 - CARD_WIDTH -
                                                   10, HEIGHT // 2 + 50)))
                        card_flip.play()

                        # Dealer first card
                        dealer_card = deal_card(deck)
                        dealer_hand.append(dealer_card)
                        card_animations.append(
                            create_deal_animation(dealer_card, deck_pos,
                                                  (WIDTH // 2 - CARD_WIDTH -
                                                   10, HEIGHT // 2 - 150)))
                        card_flip.play()

                        # Player second card
                        player_card = deal_card(deck)
                        player_hands[0].append(player_card)
                        card_animations.append(
                            create_deal_animation(
                                player_card, deck_pos,
                                (WIDTH // 2 + 10, HEIGHT // 2 + 50)))
                        card_flip.play()

                        # Dealer second card (face down)
                        dealer_card = deal_card(deck)
                        dealer_hand.append(dealer_card)
                        card_animations.append(
                            create_deal_animation(
                                dealer_card, deck_pos,
                                (WIDTH // 2 + 10, HEIGHT // 2 - 150)))
                        card_flip.play()

                # Handle insurance option
                elif game_state == "INSURANCE":
                    # Check if player wants insurance
                    if insurance_button.collidepoint(mouse_pos):
                        # Insurance costs half the original bet
                        insurance_bet = current_bet // 2
                        PLAYER_MONEY -= insurance_bet
                        save_player_money(PLAYER_MONEY)

                        # Create chip animation for insurance bet
                        chip_animations.append(
                            ChipAnimation(insurance_bet,
                                          (WIDTH // 2 - 150, HEIGHT - 150),
                                          (WIDTH // 2 - 150, HEIGHT - 300)))
                        chip_place.play()

                        # Check if dealer has a natural Blackjack
                        if calculate_hand(dealer_hand) == 21:
                            # Insurance pays 2:1
                            PLAYER_MONEY += insurance_bet * 3
                            save_player_money(PLAYER_MONEY)
                            stats["insurance_wins"] += 1
                            text_effects.append(
                                TextEffect("Insurance Win!",
                                           (WIDTH // 2, HEIGHT // 2 - 100), GREEN))
                            game_state = "GAME_OVER"
                            result = "DEALER BLACKJACK!"
                            show_dealer_cards = True
                            PLAYER_MONEY -= current_bet
                            save_player_money(PLAYER_MONEY)
                        else:
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

                    if VIP_ROOM_ACTIVE == True and game_state == "PLAYER_TURN" and len(player_hands[current_hand_index]) == 2:
                        draw_glowing_button(screen, surrender_button, "SURRENDER", WHITE, 
                                            (150, 150, 150), (100, 100, 100))
                        
                        if surrender_button.collidepoint(pygame.mouse.get_pos()):
                            tooltip_font = pygame.font.Font(None, 24)
                            tooltip_text = tooltip_font.render("Give up hand and recover half your bet", True, WHITE)
                            pygame.draw.rect(screen, (0, 0, 0, 200),
                                              (surrender_button.x, surrender_button.y - 30, 
                                              tooltip_text.get_width() + 10, 25))
                            screen.blit(tooltip_text, 
                                        (surrender_button.x + 5, surrender_button.y - 25))

                    # Hit button
                    if hit_button.collidepoint(mouse_pos):
                        player_card = deal_card(deck)
                        player_hand.append(player_card)

                        # Calculate position based on number of cards and current hand index
                        offset = len(player_hand) - 3
                        y_offset = current_hand_index * 100  # Offset for split hands
                        card_animations.append(
                            create_deal_animation(
                                player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                (WIDTH // 2 + CARD_WIDTH + offset * 30,
                                 HEIGHT // 2 + 50 + y_offset)))

                        # Check if player busts
                        if calculate_hand(player_hand) > 21:
                            if current_hand_index < len(player_hands) - 1:
                                player_bust = pygame.mixer.Sound(
                                    "assets/sounds/player_bust.wav")
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
                                    player_bust = pygame.mixer.Sound(
                                        "assets/sounds/player_bust.wav")
                                    player_bust.play()
                                    show_dealer_cards = True
                                    PLAYER_MONEY = int(PLAYER_MONEY -
                                                       current_bet)
                                    save_player_money(PLAYER_MONEY)
                                    text_effects.append(
                                        TextEffect(result,
                                                   (WIDTH // 2, HEIGHT // 2),
                                                   RED))
                                    lose = pygame.mixer.Sound(
                                        "assets/sounds/lose.wav")
                                    lose.play()
                                    particle_systems.extend(
                                        check_achievements(
                                            game_state, result, player_hand,
                                            dealer_hand, PLAYER_MONEY,
                                            current_bet, achievements_unlocked,
                                            text_effects, particle_systems,
                                            stats) or [])

                    # Stand button
                    elif stand_button.collidepoint(mouse_pos):
                        if current_hand_index < len(player_hands) - 1:
                            # If there are more split hands, move to the next one
                            split_results.append("STAND")
                            standing = pygame.mixer.Sound(
                                "assets/sounds/standing.wav")
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
                    elif double_button.collidepoint(mouse_pos) and len(
                            player_hand) == 2:
                        if PLAYER_MONEY >= current_bet:
                            PLAYER_MONEY -= current_bet

                            # Add animation for doubling chips
                            chip_animations.append(
                                ChipAnimation(current_bet,
                                              (WIDTH // 2 - 100, HEIGHT - 150),
                                              (WIDTH // 2, HEIGHT - 200)))

                            double = pygame.mixer.Sound(
                                "assets/sounds/double_down.wav")
                            double.play()

                            chip_place.play()

                            # Deal one card to player
                            player_card = deal_card(deck)
                            player_hand.append(player_card)

                            # Calculate y offset for split hands
                            y_offset = current_hand_index * 100
                            card_animations.append(
                                create_deal_animation(
                                    player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + CARD_WIDTH,
                                     HEIGHT // 2 + 50 + y_offset)))

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
                    elif split_button.collidepoint(mouse_pos) and len(
                            player_hand) == 2:
                        # Check if cards are the same value and player has enough money
                        card1_value = get_card_value(player_hand[0])
                        card2_value = get_card_value(player_hand[1])

                        if card1_value == card2_value and PLAYER_MONEY >= current_bet:
                            # Take additional bet for the split hand
                            PLAYER_MONEY -= current_bet
                            save_player_money(PLAYER_MONEY) 

                            # Create a new hand with the second card
                            new_hand = [player_hand.pop()]
                            player_hands.append(new_hand)

                            split_bets.append(current_bet)

                            split = pygame.mixer.Sound(
                                "assets/sounds/split_hand.wav")
                            split.play()

                            # Add animation for splitting chips
                            chip_animations.append(
                                ChipAnimation(
                                    current_bet,
                                    (WIDTH // 2 - 100, HEIGHT - 150),
                                    (WIDTH // 2 + 100, HEIGHT - 200)))
                            chip_place.play()

                            # Deal a new card to the first hand
                            player_card = deal_card(deck)
                            player_hand.append(player_card)
                            card_animations.append(
                                create_deal_animation(
                                    player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + 10, HEIGHT // 2 + 50)))

                            # Deal a new card to the second hand
                            second_card = deal_card(deck)
                            player_hands[1].append(second_card)
                            card_animations.append(
                                create_deal_animation(
                                    second_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + 10, HEIGHT // 2 + 150)))

                            # Record split for achievements
                            stats["splits"] += 1

                    elif surrender_button.collidepoint(mouse_pos) and VIP_ROOM_ACTIVE and len(player_hand) == 2:
                        # Surrender gives back half the bet
                        PLAYER_MONEY += current_bet // 2
                        save_player_money(PLAYER_MONEY)
                        game_state = "GAME_OVER"
                        result = "SURRENDER"
                        show_dealer_cards = True
                        text_effects.append(
                            TextEffect(result, (WIDTH // 2, HEIGHT // 2), (150, 150, 150)))
            
                        # Play surrender sound if available
                        try:
                            surrender_sound = pygame.mixer.Sound("assets/sounds/surrender.wav")
                            surrender_sound.play()
                        except:
                            pass  # Skip if sound file doesn't exist

                # Handle game over state - start new game
                elif game_state == "GAME_OVER":
                    if hit_button.collidepoint(
                            mouse_pos) or stand_button.collidepoint(mouse_pos):
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
                        deck = shuffle_deck()

        # Update animations
        for animation in card_animations[:]:
            animation.update()
            if animation.complete:
                card_animations.remove(animation)
                # If all deal animations complete, check for dealer ace and offer insurance
                if game_state == "DEALING" and not card_animations:
                    
                    # Check if dealer's face-up card is an Ace
                    dealer_first_card = dealer_hand[0]
                    if dealer_first_card[0] == 'ace':
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
                            text_effects.append(
                                TextEffect(result, (WIDTH // 2, HEIGHT // 2),
                                           BLUE))

                            # Process insurance if taken
                            if insurance_bet > 0:
                                # Insurance pays 2:1
                                PLAYER_MONEY += insurance_bet * 3
                                save_player_money(PLAYER_MONEY)
                                stats["insurance_wins"] += 1
                                text_effects.append(
                                    TextEffect("Insurance Win!",
                                               (WIDTH // 2, HEIGHT // 2 + 50),
                                               GREEN))
                        else:
                            game_state = "GAME_OVER"
                            result = "BLACKJACK!"
                            nat_blackjack = pygame.mixer.Sound(
                                "assets/sounds/nat_blackjack.wav")
                            nat_blackjack.play()
                            show_dealer_cards = True
                            # Blackjack pays 3:2
                            PLAYER_MONEY += int(current_bet * 2.5)
                            save_player_money(PLAYER_MONEY)
                            text_effects.append(
                                TextEffect(result, (WIDTH // 2, HEIGHT // 2),
                                           GOLD))

                            # Check for jackpot win
                            if JACKPOT_ENABLED and JACKPOT_TRIGGER == "natural_blackjack":
                                PLAYER_MONEY += JACKPOT_AMOUNT
                                text_effects.append(
                                    TextEffect(f"JACKPOT WON: ${JACKPOT_AMOUNT}!", (WIDTH // 2, HEIGHT // 2 - 150), GOLD))
                                JACKPOT_AMOUNT = 0  # Reset jackpot

                        particle_systems.extend(
                            check_achievements(
                                game_state, result, player_hand, dealer_hand,
                                PLAYER_MONEY, current_bet,
                                achievements_unlocked, text_effects,
                                particle_systems, stats) or [])

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
            if dealer_value <= 16:
                dealer_card = deal_card(deck)
                dealer_hand.append(dealer_card)
                card_animations.append(
                    create_deal_animation(
                        dealer_card, 
                        (WIDTH - CARD_WIDTH - 50, 50),
                        (WIDTH // 2 + len(dealer_hand) * 20, HEIGHT // 2 - 150)
                    )
                )
                card_flip.play()
            else:
                # Dealer is done hitting, determine results
                dealer_value = calculate_hand(dealer_hand)
                game_state = "GAME_OVER"
                
                # Handle results for each player hand
                if len(player_hands) > 1:
                    for i, hand in enumerate(player_hands):
                        player_value = calculate_hand(hand)
                        
                        # Skip hands that already busted
                        if i < len(split_results) and "BUST" in split_results[i]:
                            continue
                        
                        # For hands that were doubled
                        is_doubled = i < len(split_results) and "DOUBLE" in split_results[i]
                        hand_bet = current_bet * 2 if is_doubled else current_bet
                
                        if dealer_value > 21:  # Dealer busts
                            split_results[i] = "WIN"
                            PLAYER_MONEY += hand_bet * 2
                            save_player_money(PLAYER_MONEY)
                            stats["dealer_busts"] += 1
                            dealer_bust = pygame.mixer.Sound("assets/sounds/dealer_bust.wav")
                            dealer_bust.play()
                        elif player_value > dealer_value:  # Player wins
                            split_results[i] = "WIN"
                            PLAYER_MONEY += hand_bet * 2
                            save_player_money(PLAYER_MONEY)
                            win = pygame.mixer.Sound("assets/sounds/win.wav")
                            win.play()
                        elif player_value == dealer_value:  # Push
                            split_results[i] = "PUSH"
                            PLAYER_MONEY += hand_bet
                            save_player_money(PLAYER_MONEY)
                            push = pygame.mixer.Sound("assets/sounds/push.wav")
                            push.play()
                        else:  # Dealer wins
                            split_results[i] = "LOSE"
                            PLAYER_MONEY -= hand_bet
                            save_player_money(PLAYER_MONEY)
                            lose = pygame.mixer.Sound("assets/sounds/lose.wav")
                            lose.play()
                else:
                    # Single hand
                    player_hand = player_hands[0]
                    player_value = calculate_hand(player_hand)
                    
                    if dealer_value > 21:  # Dealer busts
                        result = "DEALER BUSTS!"
                        PLAYER_MONEY += current_bet * 2
                        save_player_money(PLAYER_MONEY)
                        stats["dealer_busts"] += 1
                        dealer_bust = pygame.mixer.Sound("assets/sounds/dealer_bust.wav")
                        dealer_bust.play()
                        text_effects.append(TextEffect(result, (WIDTH // 2, HEIGHT // 2), GREEN))
                    elif player_value > dealer_value:  # Player wins
                        result = "YOU WIN!"
                        PLAYER_MONEY += current_bet * 2
                        save_player_money(PLAYER_MONEY)
                        win = pygame.mixer.Sound("assets/sounds/win.wav")
                        win.play()
                        text_effects.append(TextEffect(result, (WIDTH // 2, HEIGHT // 2), GREEN))
                    elif player_value == dealer_value:  # Push
                        result = "PUSH"
                        PLAYER_MONEY += current_bet
                        save_player_money(PLAYER_MONEY)
                        push = pygame.mixer.Sound("assets/sounds/push.wav")
                        push.play()
                        text_effects.append(TextEffect(result, (WIDTH // 2, HEIGHT // 2), BLUE))
                    else:  # Dealer wins
                        result = "DEALER WINS"
                        PLAYER_MONEY -= current_bet
                        save_player_money(PLAYER_MONEY)
                        lose = pygame.mixer.Sound("assets/sounds/lose.wav")
                        lose.play()
                        text_effects.append(TextEffect(result, (WIDTH // 2, HEIGHT // 2), RED))
            
                # Check achievements
                particle_systems.extend(
                    check_achievements(
                        game_state, result, player_hand, dealer_hand,
                        PLAYER_MONEY, current_bet, achievements_unlocked,
                        text_effects, particle_systems, stats) or []
                )

        # Draw game
        screen.fill(BLACK)

        # Draw casino table background
        if VIP_ROOM_ACTIVE:
            screen.blit(VIP_CASINO_TABLE, (0, 0))
        else:
            screen.blit(CASINO_TABLE, (0, 0))

        # Draw deck
        screen.blit(CARD_BACK, (WIDTH - CARD_WIDTH - 50, 50))

        # Draw dealer's cards
        if dealer_hand:
            dealer_up_card_text = FONT.render(f"Dealer Shows: {get_card_value(dealer_hand[0])}", True, WHITE)
            for i, card in enumerate(dealer_hand):
                if i == 1 and not show_dealer_cards:
                    # Draw face down card
                    screen.blit(CARD_BACK,
                                (WIDTH // 2 + 10, HEIGHT // 2 - 150))
                else:
                    # Draw face up card
                    screen.blit(card_images[card],
                                (WIDTH // 2 - CARD_WIDTH - 10 + i *
                                 (CARD_WIDTH + 20), HEIGHT // 2 - 150))

            # Draw dealer's hand value
            if show_dealer_cards:
                value_text = FONT.render(
                    f"Dealer: {calculate_hand(dealer_hand)}", True, WHITE)
                screen.blit(value_text, (WIDTH // 2 - 60, HEIGHT // 2 - 200))

        for hand_index, hand in enumerate(player_hands):
            if hand:
                y_offset = hand_index * 100  # Offset for split hands

                for i, card in enumerate(hand):
                    screen.blit(
                        card_images[card],
                        (WIDTH // 2 - CARD_WIDTH - 10 + i *
                         (CARD_WIDTH + 20), HEIGHT // 2 + 50 + y_offset))

                # Draw hand value
                value_text = FONT.render(
                    f"Hand {hand_index + 1}: {calculate_hand(hand)}", True,
                    WHITE)
                screen.blit(
                    value_text,
                    (WIDTH // 2 - 250, HEIGHT // 2 + 80 + y_offset))

                # For split hands in game over state, show result
                if game_state == "GAME_OVER" and len(
                        player_hands) > 1 and hand_index < len(split_results):
                    result_text = FONT.render(
                        split_results[hand_index], True,
                        GREEN if split_results[hand_index] == "WIN" else
                        RED if split_results[hand_index] == "LOSE"
                        or split_results[hand_index] == "BUST!" else BLUE)
                    screen.blit(
                        result_text,
                        (WIDTH // 2 - 100, HEIGHT // 2 + 80 + y_offset)
                    )

        # Draw money and bet information
        money_text = FONT.render(f"Money: ${PLAYER_MONEY}", True, GOLD)
        screen.blit(money_text, (50, 50))

        bet_text = FONT.render(f"Current Bet: ${current_bet}", True, WHITE)
        screen.blit(bet_text, (50, 100))

        # Draw insurance bet if active
        if insurance_bet > 0:
            insurance_text = FONT.render(f"Insurance: ${insurance_bet}", True,
                                         WHITE)
            screen.blit(insurance_text, (50, 150))

        # Draw jackpot amount if enabled
        if JACKPOT_ENABLED:
            jackpot_text = FONT.render(f"Jackpot: ${JACKPOT_AMOUNT}", True, GOLD)
            screen.blit(jackpot_text, (WIDTH - 250, 100))

        # Draw buttons based on game state
        if game_state == "BETTING":
            # Draw chips for betting
            draw_chips(screen, current_bet)

            # Draw deal button if bet is valid
            if current_bet >= MIN_BET:
                draw_glowing_button(screen, deal_button, "DEAL", WHITE, GREEN,
                                    (100, 255, 100))
              
            if PLAYER_MONEY >= MIN_BET:
                draw_glowing_button(screen, all_in_button, "ALL-IN", WHITE, RED, (255, 100, 100))
            
            # Reset button to clear bet
            if current_bet > 0:
                reset_button = pygame.Rect(WIDTH // 2 - 200, HEIGHT - 100, 100,
                                           50)
                draw_glowing_button(screen, reset_button, "RESET", WHITE, RED,
                                    (255, 100, 100))

                # Handle reset button click
                if pygame.mouse.get_pressed()[0]:
                    mouse_pos = pygame.mouse.get_pos()
                    if reset_button.collidepoint(mouse_pos):
                        current_bet = 0

            # Handle All-In button click
            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if all_in_button.collidepoint(mouse_pos):
                    # Set current bet to player's total money, but not exceeding MAX_BET
                    current_bet = PLAYER_MONEY
                    logging.info(f"Player went all-in with ${current_bet}")
                    # Create chip animation for the all-in bet
                    chip_animations.append(
                ChipAnimation(current_bet, (WIDTH // 2 - 100, HEIGHT - 150), (WIDTH // 2, HEIGHT - 200))
                    )
                    chip_place.play()


        # Draw insurance option buttons
        elif game_state == "INSURANCE":
            # Draw message about insurance
            insurance_msg = FONT.render(
                "Dealer showing an Ace. Would you like to take insurance?",
                True, WHITE)
            screen.blit(
                insurance_msg,
                (WIDTH // 2 - insurance_msg.get_width() // 2, HEIGHT // 2))

            insurance_cost = FONT.render(
                f"Insurance costs: ${current_bet // 2}", True, GOLD)
            screen.blit(insurance_cost,
                        (WIDTH // 2 - insurance_cost.get_width() // 2,
                         HEIGHT // 2 + 40))

            # Draw insurance buttons
            draw_glowing_button(screen, insurance_button, "TAKE INSURANCE",
                                WHITE, GREEN, (100, 255, 100))
            draw_glowing_button(screen, no_insurance_button, "NO INSURANCE",
                                WHITE, RED, (255, 100, 100))
            
            

        elif game_state == "PLAYER_TURN":
            # Get current hand
            player_hand = player_hands[current_hand_index]

            # ====== AUTO-PLAY LOGIC ======
            if VIP_ROOM_ACTIVE and game_state in ["PLAYER_TURN", "DEALER_TURN"]:
                auto_play_rect = pygame.Rect(WIDTH - 150, 50, 120, 40)
                draw_glowing_button(
                    screen, auto_play_rect,
                    "AUTO-PLAY: ON" if vip_room.auto_play else "AUTO-PLAY: OFF",
                    BLACK, GREEN if vip_room.auto_play else RED,
                    (100, 255, 100) if vip_room.auto_play else (255, 100, 100)
                )

                # Handle auto-play toggle on click
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    current_time = pygame.time.get_ticks()

                    if auto_play_rect.collidepoint(mouse_pos):
                        vip_room.auto_play = not vip_room.auto_play
                        vip_room.last_click_time = current_time  # Update last click time

                        if vip_room.auto_play:
                            text_effects.append(TextEffect(
                                "AUTO-PLAY: ON",
                                (WIDTH//2, HEIGHT//2 + 200),
                                BLUE
                            ))
                        else:
                            text_effects.append(TextEffect(
                                "AUTO-PLAY: OFF",
                                (WIDTH//2, HEIGHT//2 + 200),
                                RED
                            ))

            # ====== AUTO-PLAY LOGIC ======
            if VIP_ROOM_ACTIVE and vip_room.auto_play and game_state == "PLAYER_TURN" and not card_animations:
                current_time = pygame.time.get_ticks()
                if current_time - vip_room.last_auto_play_time > vip_room.auto_play_delay * 1000:
                    vip_room.last_auto_play_time = current_time
                    player_hand = player_hands[current_hand_index]
                    recommended = strategy_assistant.get_recommended_move(player_hand, dealer_hand[0])

                    print(f"Auto-play recommendation: {recommended}")

                    # Simulate button clicks based on recommendation
                    if "HIT" in recommended:
                        player_card = deal_card(deck)
                        player_hand.append(player_card)
                        offset = len(player_hand) - 3
                        y_offset = current_hand_index * 100
                        card_animations.append(
                           create_deal_animation(
                                player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                (WIDTH // 2 + CARD_WIDTH + offset * 30,
                                 HEIGHT // 2 + 50 + y_offset)))
                        
                        if calculate_hand(player_hand) > 21:
                            if current_hand_index < len(player_hands) - 1:
                                # Play bust sound
                                player_bust = pygame.mixer.Sound("assets/sounds/player_bust.wav")
                                player_bust.play()

                                # Record bust for this hand and move to next split hand
                                split_results.append("BUST!")
                                current_hand_index += 1
                            else:
                                # If this is the last hand, end the player's turn
                                if len(player_hands) > 1:
                                    split_results.append("BUST!")
                                    game_state = "DEALER_TURN"
                                    show_dealer_cards = True
                                else:
                                    # Single hand - player loses
                                    game_state = "GAME_OVER"
                                    result = "BUST!"
                                    player_bust = pygame.mixer.Sound("assets/sounds/player_bust.wav")
                                    player_bust.play()
                                    show_dealer_cards = True
                                    PLAYER_MONEY -= current_bet
                                    save_player_money(PLAYER_MONEY)
                                    text_effects.append(
                                        TextEffect(result, (WIDTH // 2, HEIGHT // 2), RED))
                                    lose = pygame.mixer.Sound("assets/sounds/lose.wav")
                                    lose.play()
                
                                    # Check for achievements
                                    particle_systems.extend(
                                    check_achievements(
                                            game_state, result, player_hand,
                                            dealer_hand, PLAYER_MONEY,
                                        current_bet, achievements_unlocked,
                                        text_effects, particle_systems,
                                        stats) or [])

                    elif "STAND" in recommended:
                        # Execute stand logic
                        if current_hand_index < len(player_hands) - 1:
                            split_results.append("STAND")
                            current_hand_index += 1
                        else:
                            if len(player_hands) > 1:
                                split_results.append("STAND")
                            game_state = "DEALER_TURN"
                            show_dealer_cards = True

                    elif "DOUBLE" in recommended and len(player_hand) == 2:
                        if PLAYER_MONEY >= current_bet:
                            PLAYER_MONEY -= current_bet
                            chip_animations.append(
                                ChipAnimation(current_bet,
                                            (WIDTH // 2 - 100, HEIGHT - 150),
                                            (WIDTH // 2, HEIGHT - 200)))
                            double = pygame.mixer.Sound("assets/sounds/double_down.wav")
                            double.play()
                            chip_place.play()
                
                            # Deal one card
                            player_card = deal_card(deck)
                            player_hand.append(player_card)
                            y_offset = current_hand_index * 100
                            card_animations.append(
                                create_deal_animation(
                                    player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + CARD_WIDTH,
                                     HEIGHT // 2 + 50 + y_offset)))

                            # Move to next hand or dealer
                            if calculate_hand(player_hand) > 21:
                                if current_hand_index < len(player_hands) - 1:
                                    split_results.append("BUST!")
                                    current_hand_index += 1
                                else:
                                    if len(player_hands) > 1:
                                        split_results.append("BUST!")
                                    game_state = "DEALER_TURN"
                                    show_dealer_cards = True
                            else:
                                if current_hand_index < len(player_hands) - 1:
                                    split_results.append("DOUBLE")
                                    current_hand_index += 1
                                else:
                                    if len(player_hands) > 1:
                                        split_results.append("DOUBLE")
                                        game_state = "DEALER_TURN"
                                        show_dealer_cards = True

                    elif "SPLIT" in recommended and len(player_hand) == 2:
                        # Execute split logic
                        card1_value = get_card_value(player_hand[0])
                        card2_value = get_card_value(player_hand[1])
            
                        if card1_value == card2_value and PLAYER_MONEY >= current_bet:
                            PLAYER_MONEY -= current_bet
                            new_hand = [player_hand.pop()]
                            player_hands.append(new_hand)
                            player_card = deal_card(deck)
                            player_hand.append(player_card)
                            split = pygame.mixer.Sound("assets/sounds/split_hand.wav")
                            split.play()
                            player_card = deal_card(deck)
                            player_hand.append(player_card)
                            card_animations.append(
                                create_deal_animation(
                                    player_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + 10, HEIGHT // 2 + 50)))

                            second_card = deal_card(deck)
                            player_hands[1].append(second_card)
                            card_animations.append(
                                create_deal_animation(
                                    second_card, (WIDTH - CARD_WIDTH - 50, 50),
                                    (WIDTH // 2 + 10, HEIGHT // 2 + 150)))

                            stats["splits"] += 1
                        else:
                            if PLAYER_MONEY < current_bet:
                                text_effects.append(TextEffect("Not enough money to split!",
                                                               (WIDTH//2, HEIGHT//2), 
                                                               RED))
                            

                    elif "SURRENDER" in recommended and VIP_ROOM_ACTIVE and len(player_hand) == 2:
                        PLAYER_MONEY += current_bet // 2
                        save_player_money(PLAYER_MONEY)
                        game_state = "GAME_OVER"
                        result = "SURRENDER"
                        show_dealer_cards = True
                        text_effects.append(
                            TextEffect(result, (WIDTH // 2, HEIGHT // 2), (150, 150, 150)))
                        try:
                            surrender_sound = pygame.mixer.Sound("assets/sounds/surrender.wav")
                            surrender_sound.play()
                        except:
                            pass            
            
            # Draw indicator for current hand
            if len(player_hands) > 1:
                indicator_text = FONT.render(
                    f"Playing Hand {current_hand_index + 1}", True, GOLD)
                screen.blit(indicator_text,
                            (WIDTH // 2 - indicator_text.get_width() // 2,
                             HEIGHT // 2))

            # Draw hit and stand buttons
            draw_glowing_button(screen, hit_button, "HIT", WHITE, BLUE,
                                (100, 100, 255))
            if hit_button.collidepoint(pygame.mouse.get_pos()):
                tooltip_font = pygame.font.Font(None, 24)
                tooltip_text = tooltip_font.render("Take another card", True, WHITE)
                pygame.draw.rect(screen, (0, 0, 0, 200),
                                (hit_button.x, hit_button.y - 30, 
                                tooltip_text.get_width() + 10, 25))
                screen.blit(tooltip_text, 
                           (hit_button.x + 5, hit_button.y - 25))
            draw_glowing_button(screen, stand_button, "STAND", WHITE, RED,
                                (255, 100, 100))
            if stand_button.collidepoint(pygame.mouse.get_pos()):
                tooltip_font = pygame.font.Font(None, 24)
                tooltip_text = tooltip_font.render("Keep your current hand", True, WHITE)
                pygame.draw.rect(screen, (0, 0, 0, 200),
                                (stand_button.x, stand_button.y - 30, 
                                tooltip_text.get_width() + 10, 25))
                screen.blit(tooltip_text, 
                           (stand_button.x + 5, stand_button.y - 25))
            # Draw double button only for first decision with enough money
            if len(player_hand) == 2 and PLAYER_MONEY >= current_bet:
                draw_glowing_button(screen, double_button, "DOUBLE", WHITE,
                                    GOLD, (255, 215, 0))
                if double_button.collidepoint(pygame.mouse.get_pos()):
                    tooltip_font = pygame.font.Font(None, 24)
                    tooltip_text = tooltip_font.render("Double your bet and take exactly one more card", True, WHITE)
                    pygame.draw.rect(screen, (0, 0, 0, 200),
                                    (double_button.x, double_button.y - 30, 
                                    tooltip_text.get_width() + 10, 25))
                    screen.blit(tooltip_text, 
                               (double_button.x + 5, double_button.y - 25))

            # Draw split button only if first two cards are the same value
            if len(player_hand) == 2 and PLAYER_MONEY >= current_bet:
                card1_value = get_card_value(player_hand[0])
                card2_value = get_card_value(player_hand[1])

                if card1_value == card2_value:
                    draw_glowing_button(screen, split_button, "SPLIT", WHITE,
                                        PURPLE, (200, 100, 255))
                    if split_button.collidepoint(pygame.mouse.get_pos()):
                        tooltip_font = pygame.font.Font(None, 24)
                        tooltip_text = tooltip_font.render("Split your pair into two separate hands", True, WHITE)
                        pygame.draw.rect(screen, (0, 0, 0, 200),
                                        (split_button.x, split_button.y - 30, 
                                        tooltip_text.get_width() + 10, 25))
                        screen.blit(tooltip_text, 
                                   (split_button.x + 5, split_button.y - 25))

            # Show Surrender button only in VIP room on first decision
            if VIP_ROOM_ACTIVE and len(player_hand) == 2:
                draw_glowing_button(screen, surrender_button, "SURRENDER", WHITE, 
                                  SURRENDER_COLOR, (150, 150, 150))
        
                # Draw tooltip when hovering
                if surrender_button.collidepoint(pygame.mouse.get_pos()):
                    tooltip_font = pygame.font.Font(None, 24)
                    tooltip_text = tooltip_font.render("Give up hand and recover half your bet", True, WHITE)
                    pygame.draw.rect(screen, (0, 0, 0, 200),
                                    (surrender_button.x, surrender_button.y - 30, 
                                    tooltip_text.get_width() + 10, 25))
                    screen.blit(tooltip_text, 
                               (surrender_button.x + 5, surrender_button.y - 25))

        elif game_state == "GAME_OVER":
            # Draw new hand button
            draw_glowing_button(screen, hit_button, "NEW HAND", WHITE, GREEN,
                                (100, 100, 225))

            # Draw main menu button
            main_menu_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 50, 200,
                                           40)
            draw_glowing_button(screen, main_menu_button, "MAIN MENU", WHITE,
                                BLUE, (100, 100, 255))

            # Handle main menu button click
            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                if main_menu_button.collidepoint(mouse_pos):
                    return  

            # Draw result text
            result_font = pygame.font.Font(None, 72)
            result_text = result_font.render(
                str(result) if result else "", True,  # Convert to string and handle empty case
                GOLD if "WIN" in result or "BLACKJACK" in result else
                RED if "BUST" in result or "DEALER WINS" in result else 
                (150, 150, 150) if "SURRENDER" in result else BLUE)
            screen.blit(
                result_text,
                (WIDTH // 2 - result_text.get_width() // 2, HEIGHT // 2 - 50))

            if "SURRENDER" in result:
                loss_text = FONT.render(f"- ${current_bet // 2} (Surrender)", True, (150, 150, 150))
                screen.blit(loss_text,
                           (WIDTH // 2 - loss_text.get_width() // 2,
                            HEIGHT // 2 + 20))
            elif len(player_hands) == 1:
                if "WIN" in result or "BLACKJACK" in result or "DEALER BUSTS" in result:
                    winnings = int(
                        current_bet *
                        1.5) if "BLACKJACK" in result else current_bet
                    winnings_text = FONT.render(f"+ ${winnings}", True, GREEN)
                    screen.blit(winnings_text,
                                (WIDTH // 2 - winnings_text.get_width() // 2,
                                 HEIGHT // 2 + 20))
                elif "PUSH" in result:
                    push_text = FONT.render("Bet Returned", True, BLUE)
                    screen.blit(push_text,
                                (WIDTH // 2 - push_text.get_width() // 2,
                                 HEIGHT // 2 + 20))
                else:
                    loss_text = FONT.render(f"- ${current_bet}", True, RED)
                    screen.blit(loss_text,
                                (WIDTH // 2 - loss_text.get_width() // 2,
                                 HEIGHT // 2 + 20))

        # Draw current animations on top of everything
        for animation in card_animations:
            card = animation.card
            if animation.complete:
                continue

            is_dealer_face_down = False
            if card in dealer_hand and dealer_hand.index(
                    card) == 1 and not show_dealer_cards:
                is_dealer_face_down = True

            if is_dealer_face_down:
                animation.draw(screen, card_images[card], CARD_BACK)
            else:
                animation.draw(screen, card_images[card], CARD_BACK)

        # Draw chip animations
        for animation in chip_animations:
            if animation.complete:
                continue

            chip_img = chip_images.get(animation.value, placeholder_chip)
            screen.blit(chip_img, animation.current_pos)

        # Draw text effects
        for effect in text_effects:
            effect.draw(screen)

        # Draw particle systems
        for system in particle_systems:
            system.draw(screen)

        # Draw achievements notification if any were unlocked this hand
        if achievements_unlocked:
            achievements_area = pygame.Rect(
                WIDTH - 300, 150, 250, 100 + len(achievements_unlocked) * 40)
            pygame.draw.rect(screen, (0, 0, 0, 180),
                             achievements_area,
                             border_radius=10)
            pygame.draw.rect(screen,
                             GOLD,
                             achievements_area,
                             2,
                             border_radius=10)

            header_font = pygame.font.Font(None, 28)
            header_text = header_font.render("Recent Achievements", True, GOLD)
            screen.blit(header_text,
                        (WIDTH - 300 +
                         (250 - header_text.get_width()) // 2, 160))

            for i, key in enumerate(achievements_unlocked):
                achievement_font = pygame.font.Font(None, 24)
                ach_text = achievement_font.render(ACHIEVEMENTS[key]["name"],
                                                   True, WHITE)
                screen.blit(ach_text, (WIDTH - 280, 200 + i * 40))

        if not deck:
            deck = shuffle_deck()  # Reshuffle if the deck is empty

        if PLAYER_MONEY <= 0:
            # Show game over screen
            choice = game_over_screen()
            if choice == "RESTART":
                # Reset player money and restart the game
                PLAYER_MONEY = 1000  # Reset to default starting money
                VIP_ROOM_ACTIVE = False
                save_player_money(PLAYER_MONEY)
                return  # Exit the current game loop and restart
            else:
                # Quit the game
                running = False
                PLAYER_MONEY = 1000

        # Display FPS
        fps_text = FONT.render(f"FPS: {int(clock.get_fps())}", True, WHITE)
        screen.blit(fps_text, (WIDTH - 150, 20))

        if dealer_hand and player_hands:
            dealer_up_card = dealer_hand[0] if dealer_hand else None
            current_player_hand = player_hands[current_hand_index] if player_hands else None
            strategy_assistant.draw(screen, current_player_hand, dealer_up_card)

        # Update the display
        pygame.display.flip()
        clock.tick(60)

    # Save player money at the end of the game
    save_player_money(PLAYER_MONEY)

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
            VIP_ROOM_ACTIVE = False
            # Exit the game
            break
    
    pygame.quit()
