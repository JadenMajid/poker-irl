"""
watch_agents.py
---------------
Interactive Pygame visualizer for the IRL Poker Agents.

Controls:
  [SPACE] - Step forward one action
  [N]     - Skip to the next hand
  [A]     - Toggle Auto-play
"""

import os
import json
import time
import pygame
import torch
from typing import Dict, Tuple

# Import your existing game logic
from cards import Card
from game_state import GameState, Action, ActionType, NUM_PLAYERS
from poker_env import PokerEnv
from agent import ActorCriticNetwork
from feature_encoder import FEATURE_DIM
from step2_train_perturbed_agents import FINETUNE_PPO_CFG
from reward import RewardFunction, RewardParams
from step5a_train_ablation_agent import AdaptiveAgent

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cpu")
HIDDEN_DIM = 256

# --- Colors ---
BG_COLOR = (30, 30, 30)
TABLE_COLOR = (34, 139, 34)
TABLE_OUTLINE = (20, 100, 20)
TEXT_COLOR = (255, 255, 255)
CARD_BG = (250, 250, 250)
RED_SUIT = (220, 20, 20)
BLACK_SUIT = (20, 20, 20)
GOLD = (255, 215, 0)


class PokerVisualizer:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1000, 750
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Multi-Agent Poker IRL Visualizer")
        
        self.font_main = pygame.font.SysFont("segoeui", 20)
        self.font_large = pygame.font.SysFont("segoeui", 28, bold=True)
        self.font_card_rank = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_card_suit = pygame.font.SysFont("segoeui", 24)

        self.env = None
        self.agent_params = {}
        self.callbacks = []
        
        self.state_generator = None
        self.current_state: GameState = None
        self.last_action: Action = None
        
        self.auto_play = False
        self.last_step_time = 0

        self.load_agents()
        self.start_new_hand()

    def load_agents(self):
        """Load the PyTorch models and their reward parameters."""
        print("Loading agents...")
        self.agents = [] # Keep a list of actual agent objects
        params_path = os.path.join(CHECKPOINT_DIR, "perturbed_agent_params.json")
        with open(params_path, "r") as f:
            params_list = json.load(f)
            for p in params_list:
                self.agent_params[p["seat"]] = (p["alpha"], p["beta"])

        for seat in range(NUM_PLAYERS):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"perturbed_agent_{seat}.pt")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            
            net = ActorCriticNetwork(
                input_dim=ckpt.get("feature_dim", FEATURE_DIM),
                hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM)
            ).to(DEVICE)
            net.load_state_dict(ckpt["network_state"])
            net.eval()

            # Create a lightweight wrapper just to get the .act() method
            # We use 0.0 for parameters here because we just need the forward pass
            rf = RewardFunction(RewardParams(0, 0)) 
            agent = AdaptiveAgent(seat, net, net, rf, FINETUNE_PPO_CFG, DEVICE)
            self.agents.append(agent)
            self.callbacks.append(agent.act)

        self.env = PokerEnv(self.callbacks, record_trajectories=False)

    def hand_step_generator(self):
        self.env._hand_number += 1
        
        # Now use the agent instances directly
        for agent in self.agents:
            agent.begin_hand()
        
        state = self.env._initialize_hand()
        yield state, None  # Initial deal

        while not state.is_terminal:
            seat = state.current_player
            obs = state.observation_for_player(seat)
            action = self.env._callbacks[seat](obs)
            action = self.env._validate_and_coerce(action, state, seat)
            
            self.env._apply_action(state, action)
            yield state, action
            
        # Hand is over
        chip_deltas = self.env._resolve_terminal(state)
        for pid, delta in chip_deltas.items():
            self.env._stacks[pid] += delta

    def start_new_hand(self):
        self.state_generator = self.hand_step_generator()
        self.step_hand()

    def step_hand(self):
        try:
            self.current_state, self.last_action = next(self.state_generator)
        except StopIteration:
            self.start_new_hand()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            current_time = time.time()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.step_hand()
                    elif event.key == pygame.K_n:
                        self.start_new_hand()
                    elif event.key == pygame.K_a:
                        self.auto_play = not self.auto_play

            # Auto-play logic
            if self.auto_play and current_time - self.last_step_time > 0.8:
                self.step_hand()
                self.last_step_time = current_time

            self.render()
            pygame.display.flip()
            clock.tick(30)
            
        pygame.quit()

    # --- Rendering Methods ---

    def render(self):
        self.screen.fill(BG_COLOR)
        
        # Draw Table
        table_rect = pygame.Rect(150, 150, 700, 450)
        pygame.draw.ellipse(self.screen, TABLE_OUTLINE, table_rect.inflate(10, 10))
        pygame.draw.ellipse(self.screen, TABLE_COLOR, table_rect)

        if not self.current_state:
            return

        self.draw_board_and_pot()
        self.draw_players()
        self.draw_ui_overlay()

    def draw_board_and_pot(self):
        state = self.current_state
        
        # Pot
        pot_text = self.font_large.render(f"Pot: {state.pot}", True, GOLD)
        self.screen.blit(pot_text, (self.width//2 - pot_text.get_width()//2, self.height//2 - 60))
        
        # Board Cards
        start_x = self.width//2 - (len(state.board_cards) * 30)
        for i, card in enumerate(state.board_cards):
            self.draw_card(card, start_x + (i * 60), self.height//2 - 10)

    def draw_players(self):
        # Top, Right, Bottom, Left
        positions = [
            (self.width//2, 80),
            (self.width - 150, self.height//2),
            (self.width//2, self.height - 100),
            (150, self.height//2)
        ]

        state = self.current_state
        
        for seat in range(NUM_PLAYERS):
            player = state.players[seat]
            x, y = positions[seat]
            alpha, beta = self.agent_params.get(seat, (0.0, 0.0))
            
            is_active = player.is_active
            is_turn = state.current_player == seat and not state.is_terminal

            # Draw Character
            self.draw_character(x, y, alpha, beta, is_active, is_turn)

            # Draw Info (Stack, Bet)
            info_y = y + 45
            txt_color = TEXT_COLOR if is_active else (100, 100, 100)
            
            name_txt = self.font_main.render(f"Seat {seat} (Stack: {player.stack})", True, txt_color)
            self.screen.blit(name_txt, (x - name_txt.get_width()//2, info_y))
            
            if player.street_investment > 0:
                bet_txt = self.font_main.render(f"Bet: {player.street_investment}", True, GOLD)
                self.screen.blit(bet_txt, (x - bet_txt.get_width()//2, info_y + 20))

            # Draw Hole Cards
            if is_active or state.is_terminal:
                for i, card in enumerate(player.hole_cards):
                    cx = x - 30 + (i * 35)
                    cy = y - 30 if seat == 2 else y - 30 # Adjust based on seat if desired
                    self.draw_card(card, cx, cy - 60)

    def draw_character(self, x, y, alpha, beta, is_active, is_turn):
        # Base head
        head_color = (200, 180, 160) if is_active else (80, 80, 80)
        if is_turn:
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 34, 3) # Highlight turn
        pygame.draw.circle(self.screen, head_color, (x, y), 30)

        if not is_active:
            return # Dead men tell no tales

        # Alpha Feature (Risk Aversion)
        if alpha > 0:
            # Risk-averse: Reading glasses & buck teeth
            pygame.draw.circle(self.screen, (50, 150, 255), (x - 10, y - 5), 10, 2)
            pygame.draw.circle(self.screen, (50, 150, 255), (x + 10, y - 5), 10, 2)
            pygame.draw.line(self.screen, (50, 150, 255), (x - 2, y - 5), (x + 2, y - 5), 2)
            # Buck teeth
            pygame.draw.rect(self.screen, (255, 255, 255), (x - 4, y + 10, 4, 6))
            pygame.draw.rect(self.screen, (255, 255, 255), (x, y + 10, 4, 6))
        else:
            # Risk-seeking: Cool Shades & smirk
            pygame.draw.rect(self.screen, (20, 20, 20), (x - 18, y - 10, 16, 12))
            pygame.draw.rect(self.screen, (20, 20, 20), (x + 2, y - 10, 16, 12))
            pygame.draw.line(self.screen, (20, 20, 20), (x - 2, y - 6), (x + 2, y - 6), 3)
            # Smirk
            pygame.draw.arc(self.screen, (0, 0, 0), (x - 10, y + 5, 20, 10), 3.14, 0, 2)

        # Beta Feature (Pot Hunger)
        if beta > 0:
            # Pot Hungry: Gold Chain
            pygame.draw.arc(self.screen, GOLD, (x - 20, y + 15, 40, 30), 3.14, 0, 3)
            chain_txt = self.font_main.render("$", True, GOLD)
            self.screen.blit(chain_txt, (x - chain_txt.get_width()//2, y + 25))
        else:
            # Pot Avoidant: Modest collar
            pygame.draw.line(self.screen, (255, 255, 255), (x - 15, y + 25), (x, y + 35), 3)
            pygame.draw.line(self.screen, (255, 255, 255), (x + 15, y + 25), (x, y + 35), 3)

    def draw_card(self, card, x, y):
        rect = pygame.Rect(x, y, 45, 65)
        pygame.draw.rect(self.screen, CARD_BG, rect, border_radius=4)
        pygame.draw.rect(self.screen, (100, 100, 100), rect, 1, border_radius=4)

        color = RED_SUIT if card.suit.name in ("HEARTS", "DIAMONDS") else BLACK_SUIT
        
        # Needs to be converted to str based on your cards.py enum mapping
        rank_str = str(card.rank) 
        suit_str = card.suit.symbol()

        r_surf = self.font_card_rank.render(rank_str, True, color)
        s_surf = self.font_card_suit.render(suit_str, True, color)

        self.screen.blit(r_surf, (x + 3, y + 2))
        self.screen.blit(s_surf, (x + 25, y + 35))

    def draw_ui_overlay(self):
        # Controls info
        controls = "Controls: [SPACE] Step  |  [N] Next Hand  |  [A] Auto-Play"
        auto_status = "ON" if self.auto_play else "OFF"
        ctrl_surf = self.font_main.render(f"{controls} (Auto: {auto_status})", True, (150, 150, 150))
        self.screen.blit(ctrl_surf, (10, 10))

        # Hand & Street Info
        state = self.current_state
        info = f"Hand: {state.hand_number} | Street: {state.street.name}"
        if state.is_terminal:
            winners = ", ".join([str(w) for w in state.winners])
            info += f" | GAME OVER! Winner(s): {winners}"
        
        info_surf = self.font_large.render(info, True, TEXT_COLOR)
        self.screen.blit(info_surf, (10, 40))

        # Action Log
        if self.last_action:
            action_str = f"Seat {self.last_action.player_id} did: {self.last_action.action_type.name}"
            if self.last_action.action_type == ActionType.RAISE:
                action_str += f" +{self.last_action.raise_amount}"
            act_surf = self.font_large.render(action_str, True, (100, 255, 100))
            self.screen.blit(act_surf, (self.width//2 - act_surf.get_width()//2, self.height - 40))


if __name__ == "__main__":
    visualizer = PokerVisualizer()
    visualizer.run()