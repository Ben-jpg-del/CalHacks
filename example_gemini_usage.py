"""
Practical Examples of Using Gemini API for Prompt Generation

This script shows real-world use cases for the Gemini API
in the context of the Fire & Water RL game project.
"""

from gemini_prompt_generator import GeminiPromptGenerator
import time


def generate_level_descriptions(generator, num_levels=5):
    """Generate creative level descriptions for the game"""
    print("\n" + "=" * 60)
    print("1. GENERATING LEVEL DESCRIPTIONS")
    print("=" * 60)

    prompt = f"""Generate {num_levels} creative level descriptions for a cooperative puzzle platformer game
featuring Fire and Water characters with different abilities:

Fire abilities:
- Can walk through lava
- Dies in water
- Can activate red pressure plates

Water abilities:
- Can walk through water
- Dies in lava
- Can activate blue pressure plates

Requirements for each level:
- Must require cooperation between both characters
- Should have unique mechanics or layouts
- Include platforms, hazards, pressure plates, and gates
- Be challenging but solvable

Format each level as:
Level [number]: [Name]
Description: [2-3 sentences]
Key mechanic: [unique feature]

Generate {num_levels} levels now:"""

    response = generator.generate(prompt, temperature=0.8)
    print(f"\n{response}\n")
    return response


def generate_reward_function_ideas(generator):
    """Generate ideas for RL reward shaping"""
    print("\n" + "=" * 60)
    print("2. GENERATING REWARD FUNCTION IDEAS")
    print("=" * 60)

    prompt = """I'm training reinforcement learning agents for a cooperative puzzle game with Fire and Water characters.

Current reward structure:
- Reaching exit: +100
- Activating pressure plate: +20
- Death: -50

Suggest 5 alternative reward shaping strategies to improve learning, considering:
- Sparse vs dense rewards
- Milestone-based rewards
- Curiosity/exploration bonuses
- Cooperative behavior incentives
- Step penalties

Format each suggestion with:
- Name
- Description (2-3 sentences)
- Expected benefits
- Potential drawbacks"""

    response = generator.generate(prompt, temperature=0.7)
    print(f"\n{response}\n")
    return response


def analyze_training_performance(generator, episode_data):
    """Use Gemini to analyze training data and suggest improvements"""
    print("\n" + "=" * 60)
    print("3. ANALYZING TRAINING PERFORMANCE")
    print("=" * 60)

    prompt = f"""I'm training RL agents for a cooperative game. Here's the training data:

{episode_data}

Analyze this data and:
1. Identify any concerning patterns
2. Suggest hyperparameter adjustments
3. Recommend training strategy changes
4. Predict if the agents will learn to cooperate

Be specific and technical."""

    response = generator.generate(prompt, temperature=0.5)
    print(f"\n{response}\n")
    return response


def generate_test_scenarios(generator):
    """Generate diverse test scenarios for evaluating trained agents"""
    print("\n" + "=" * 60)
    print("4. GENERATING TEST SCENARIOS")
    print("=" * 60)

    prompt = """Generate 5 challenging test scenarios to evaluate trained Fire and Water RL agents.

Each scenario should test different aspects:
- Cooperation requirements
- Long-term planning
- Handling hazards
- Timing and coordination
- Edge cases

Format each as:
Scenario [number]: [Name]
Setup: [Description]
Success criteria: [What agents must do]
Tests: [What skill this evaluates]

Generate 5 scenarios:"""

    response = generator.generate(prompt, temperature=0.8)
    print(f"\n{response}\n")
    return response


def brainstorm_curriculum_learning(generator):
    """Design a curriculum learning progression"""
    print("\n" + "=" * 60)
    print("5. CURRICULUM LEARNING DESIGN")
    print("=" * 60)

    prompt = """Design a curriculum learning progression for training cooperative Fire and Water agents.

Current setup:
- Tutorial level (simple, single room)
- Tower level (complex, multi-level vertical)

Create a 6-stage curriculum from easiest to hardest:
- Stage 1: [Description]
- Stage 2: [Description]
... etc

Each stage should:
- Build on previous stages
- Introduce 1-2 new concepts
- Have clear success criteria
- Prepare for the next stage

Design the curriculum:"""

    response = generator.generate(prompt, temperature=0.7)
    print(f"\n{response}\n")
    return response


def interactive_prompt_refinement(generator):
    """Interactive session to refine prompts"""
    print("\n" + "=" * 60)
    print("6. INTERACTIVE PROMPT REFINEMENT")
    print("=" * 60)

    print("\nLet's interactively design a new level!")
    print("Enter your requirements (or press Enter to use default):\n")

    # Get user input
    theme = input("Theme (e.g., 'volcano', 'ice cave'): ").strip() or "ancient temple"
    difficulty = input("Difficulty (easy/medium/hard): ").strip() or "medium"
    focus = input("Focus (exploration/combat/puzzles): ").strip() or "puzzles"

    prompt = f"""Design a detailed {difficulty} difficulty level for a Fire and Water cooperative game.

Theme: {theme}
Focus: {focus}

Include:
1. Level layout (platforms, dimensions)
2. Hazard placement (lava pools, water pools)
3. Interactive elements (pressure plates, gates, bridges)
4. Spawn points and exit locations
5. Solution strategy (how players should cooperate)
6. Estimated completion time

Provide specific coordinates and measurements."""

    print(f"\nGenerating level design for '{theme}'...\n")
    response = generator.generate(prompt, temperature=0.8)
    print(f"{response}\n")
    return response


def main():
    """Run all examples"""
    print("=" * 60)
    print("Gemini API Prompt Generator - Practical Examples")
    print("=" * 60)

    # Initialize generator
    try:
        generator = GeminiPromptGenerator()
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\nSet your API key first:")
        print("  Windows: set GEMINI_API_KEY=your-api-key")
        print("  Linux/Mac: export GEMINI_API_KEY=your-api-key")
        return

    # Run examples
    try:
        # Example 1: Level descriptions
        generate_level_descriptions(generator, num_levels=3)
        time.sleep(1)  # Rate limiting

        # Example 2: Reward function ideas
        generate_reward_function_ideas(generator)
        time.sleep(1)

        # Example 3: Training analysis
        sample_data = """Episode 100: Reward=15.2, Success=0%, Steps=287
Episode 200: Reward=28.5, Success=5%, Steps=241
Episode 300: Reward=42.1, Success=12%, Steps=198
Episode 400: Reward=38.9, Success=8%, Steps=215
Episode 500: Reward=55.3, Success=18%, Steps=176"""

        analyze_training_performance(generator, sample_data)
        time.sleep(1)

        # Example 4: Test scenarios
        generate_test_scenarios(generator)
        time.sleep(1)

        # Example 5: Curriculum learning
        brainstorm_curriculum_learning(generator)
        time.sleep(1)

        # Example 6: Interactive (optional)
        print("\n" + "=" * 60)
        print("Would you like to try interactive level design? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            interactive_prompt_refinement(generator)

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Your API key is valid")
        print("  2. You have internet connection")
        print("  3. You haven't exceeded rate limits")


if __name__ == "__main__":
    # Check if package is installed
    try:
        import google.generativeai
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("\nInstall with:")
        print("  pip install google-generativeai")
        exit(1)

    main()
