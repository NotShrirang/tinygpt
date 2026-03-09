import torch
import argparse
import os
import readline

from tinygpt import TinyGPT2, TinyGPT2Config, TinyGPT2_1Config, Tokenizer


# ── Prompt formatting ────────────────────────────────────────────────

def format_prompt(instruction, input_text="", system=""):
    parts = []
    if system.strip():
        parts.append(f"### System:\n{system}")
    parts.append(f"### Instruction:\n{instruction}")
    if input_text.strip():
        parts.append(f"### Input:\n{input_text}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


def format_chat_prompt(messages, system=""):
    """Build a multi-turn chat prompt from message history.

    Each message is a dict with 'role' ('user' or 'assistant') and 'content'.
    The full conversation is packed into a single prompt so the model sees context.
    """
    parts = []
    if system.strip():
        parts.append(f"### System:\n{system}")
    for msg in messages:
        if msg['role'] == 'user':
            parts.append(f"### Instruction:\n{msg['content']}")
        elif msg['role'] == 'assistant':
            parts.append(f"### Response:\n{msg['content']}")
    parts.append("### Response:\n")
    return "\n\n".join(parts)


# ── Model loading ────────────────────────────────────────────────────

def get_model_config(config_name):
    if config_name == "v2.1":
        return TinyGPT2_1Config()
    return TinyGPT2Config()


def load_model(checkpoint_path, device, config_name="default"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = get_model_config(config_name)
    saved_config = checkpoint.get('config', {})
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)

    tokenizer = Tokenizer()
    model = TinyGPT2(config, pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    step = checkpoint.get('opt_step', checkpoint.get('global_step', '?'))
    train_loss = checkpoint.get('train_loss', '?')
    val_loss = checkpoint.get('val_loss', '?')
    tokens = checkpoint.get('total_tokens_processed', '?')

    info = {
        'checkpoint': os.path.basename(checkpoint_path),
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'tokens': tokens,
        'params': n_params,
    }

    return model, tokenizer, config, info


# ── Generation ───────────────────────────────────────────────────────

def generate_text(model, tokenizer, device, prompt, max_tokens, temperature, top_k,
                  raw=False, system="", stream=True):
    if raw:
        full_prompt = prompt
    else:
        full_prompt = format_prompt(prompt, system=system)

    prompt_tokens = tokenizer.encode(full_prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    if raw:
        print(prompt, end="", flush=True)

    with torch.inference_mode():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
            stream=stream,
            eos_token_id=tokenizer.eos_id,
        )
    print()

    return tokenizer.decode(generated[0].tolist())


def generate_chat(model, tokenizer, device, messages, max_tokens, temperature, top_k,
                  system="", block_size=512):
    """Generate a response given the full chat history. Truncates from the front if needed."""
    full_prompt = format_chat_prompt(messages, system=system)
    prompt_tokens = tokenizer.encode(full_prompt)

    # Truncate from the front if the context is too long, keeping room for generation
    max_prompt_len = block_size - max_tokens
    if max_prompt_len < 1:
        max_prompt_len = block_size // 2
    if len(prompt_tokens) > max_prompt_len:
        prompt_tokens = prompt_tokens[-max_prompt_len:]

    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    with torch.inference_mode():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
            stream=True,
            eos_token_id=tokenizer.eos_id,
        )
    print()

    # Extract only the new tokens (after the prompt)
    all_tokens = generated[0].tolist()
    response_tokens = all_tokens[len(prompt_tokens):]
    return tokenizer.decode(response_tokens)


# ── Slash commands ───────────────────────────────────────────────────

COMMANDS = {}


def command(name, description, usage=None):
    """Decorator to register a slash command."""
    def decorator(func):
        COMMANDS[name] = {'func': func, 'desc': description, 'usage': usage or f"/{name}"}
        return func
    return decorator


class SessionState:
    """Mutable state for the interactive session."""
    MODES = ['chat', 'instruction', 'raw']

    def __init__(self, args, model, tokenizer, config, info, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.info = info
        self.device = device
        self.max_tokens = args.max_tokens
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.system = args.system or ""
        self.history = []       # list of (prompt, response) for /history
        self.chat_messages = [] # list of {'role': 'user'|'assistant', 'content': str}
        # Mode: chat (default), instruction, raw
        if args.raw:
            self.mode = 'raw'
        elif args.mode:
            self.mode = args.mode
        else:
            self.mode = 'chat'


@command("help", "Show available commands")
def cmd_help(state, arg):
    print("\n  Available commands:\n")
    for name, cmd in sorted(COMMANDS.items()):
        print(f"    {cmd['usage']:<28} {cmd['desc']}")
    print()


@command("info", "Show model and session info")
def cmd_info(state, arg):
    info = state.info
    tokens = info['tokens']
    tokens_str = f"{tokens:,}" if isinstance(tokens, int) else str(tokens)
    print(f"\n  Model Info:")
    print(f"    Checkpoint:   {info['checkpoint']}")
    print(f"    Step:         {info['step']}")
    print(f"    Train Loss:   {info['train_loss']}")
    print(f"    Val Loss:     {info['val_loss']}")
    print(f"    Tokens seen:  {tokens_str}")
    print(f"    Parameters:   {info['params'] / 1e6:.2f}M")
    print(f"    Device:       {state.device}")
    print(f"\n  Session Settings:")
    print(f"    Mode:         {state.mode}")
    print(f"    Chat turns:   {len(state.chat_messages)}")
    print(f"    Max tokens:   {state.max_tokens}")
    print(f"    Temperature:  {state.temperature}")
    print(f"    Top-k:        {state.top_k}")
    print(f"    System:       {state.system or '(none)'}")
    print()


@command("mode", "Switch mode: chat, instruction, raw", usage="/mode <name>")
def cmd_mode(state, arg):
    if not arg:
        print(f"  Current mode: {state.mode}")
        print(f"  Available: {', '.join(SessionState.MODES)}")
        print(f"  Usage: /mode chat\n")
        return
    name = arg.strip().lower()
    if name not in SessionState.MODES:
        print(f"  Unknown mode: {name}")
        print(f"  Available: {', '.join(SessionState.MODES)}\n")
        return
    state.mode = name
    if name == 'chat':
        print(f"  Switched to chat mode (multi-turn, remembers context).\n")
    elif name == 'instruction':
        print(f"  Switched to instruction mode (single-turn, no memory).\n")
    else:
        print(f"  Switched to raw mode (no template, no memory).\n")


@command("temp", "Set temperature", usage="/temp <value>")
def cmd_temp(state, arg):
    if not arg:
        print(f"  Current temperature: {state.temperature}")
        print(f"  Usage: /temp 0.7\n")
        return
    try:
        val = float(arg)
        if val < 0:
            raise ValueError
        state.temperature = val
        print(f"  Temperature set to {val}\n")
    except ValueError:
        print(f"  Invalid value. Usage: /temp 0.7\n")


@command("topk", "Set top-k sampling", usage="/topk <value>")
def cmd_topk(state, arg):
    if not arg:
        print(f"  Current top-k: {state.top_k}")
        print(f"  Usage: /topk 50\n")
        return
    try:
        val = int(arg)
        if val < 0:
            raise ValueError
        state.top_k = val
        print(f"  Top-k set to {val}\n")
    except ValueError:
        print(f"  Invalid value. Usage: /topk 50\n")


@command("max", "Set max tokens to generate", usage="/max <value>")
def cmd_max(state, arg):
    if not arg:
        print(f"  Current max tokens: {state.max_tokens}")
        print(f"  Usage: /max 300\n")
        return
    try:
        val = int(arg)
        if val < 1:
            raise ValueError
        state.max_tokens = val
        print(f"  Max tokens set to {val}\n")
    except ValueError:
        print(f"  Invalid value. Usage: /max 300\n")


@command("system", "Set/clear system prompt", usage="/system [text]")
def cmd_system(state, arg):
    if not arg:
        if state.system:
            print(f"  Current system prompt: {state.system}")
            print(f"  Use '/system clear' to remove it.\n")
        else:
            print(f"  No system prompt set.")
            print(f"  Usage: /system You are a helpful assistant.\n")
        return
    if arg.strip().lower() == "clear":
        state.system = ""
        print(f"  System prompt cleared.\n")
    else:
        state.system = arg
        print(f"  System prompt set to: {arg}\n")


@command("history", "Show conversation history")
def cmd_history(state, arg):
    if not state.history:
        print("  No history yet.\n")
        return
    print(f"\n  History ({len(state.history)} messages):\n")
    for i, (prompt, _) in enumerate(state.history, 1):
        display = prompt if len(prompt) <= 60 else prompt[:57] + "..."
        print(f"    {i}. {display}")
    print()


@command("clear", "Clear conversation history and chat context")
def cmd_clear(state, arg):
    state.history.clear()
    state.chat_messages.clear()
    print("  History and chat context cleared.\n")


@command("new", "Start a new chat (clears chat context)")
def cmd_new(state, arg):
    state.chat_messages.clear()
    print("  New chat started.\n")


@command("last", "Repeat the last generation")
def cmd_last(state, arg):
    if not state.history:
        print("  No history yet.\n")
        return
    prompt, response = state.history[-1]
    print(f"  Prompt: {prompt}\n")
    print(response)
    print()


@command("load", "Load a different checkpoint", usage="/load <path>")
def cmd_load(state, arg):
    if not arg:
        print(f"  Current: {state.info['checkpoint']}")
        print(f"  Usage: /load path/to/checkpoint.pth\n")
        return
    path = arg.strip()
    if not os.path.exists(path):
        print(f"  File not found: {path}\n")
        return
    try:
        config_name = "v2.1" if "v2.1" in path else "default"
        model, tokenizer, config, info = load_model(path, state.device, config_name)
        state.model = model
        state.tokenizer = tokenizer
        state.config = config
        state.info = info
        print(f"  Loaded: {info['checkpoint']} ({info['params'] / 1e6:.2f}M params, step {info['step']})\n")
    except Exception as e:
        print(f"  Failed to load: {e}\n")


@command("quit", "Exit the program")
def cmd_quit(state, arg):
    raise SystemExit


# ── Interactive REPL ─────────────────────────────────────────────────

def setup_readline():
    """Configure readline for history and editing."""
    history_file = os.path.expanduser("~/.tinygpt_history")

    # Load previous history
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass

    readline.set_history_length(1000)

    # Tab completion for slash commands
    def completer(text, state):
        if text.startswith("/"):
            cmd_text = text[1:]
            options = [f"/{name}" for name in COMMANDS if name.startswith(cmd_text)]
            if state < len(options):
                return options[state]
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    return history_file


def print_banner(info, mode):
    n_params = info['params']
    tokens = info['tokens']
    tokens_str = f"{tokens:,}" if isinstance(tokens, int) else str(tokens)

    print(f"\n{'='*50}")
    print(f"  TinyGPT Inference")
    print(f"{'='*50}")
    print(f"  Checkpoint:  {info['checkpoint']}")
    print(f"  Step:        {info['step']}")
    print(f"  Train Loss:  {info['train_loss']}")
    print(f"  Val Loss:    {info['val_loss']}")
    print(f"  Tokens seen: {tokens_str}")
    print(f"  Parameters:  {n_params / 1e6:.2f}M")
    print(f"{'='*50}")
    print(f"\n  Mode: {mode} | Type /help for commands\n")


def interactive_loop(state):
    history_file = setup_readline()

    print_banner(state.info, state.mode)

    try:
        while True:
            try:
                prompt = input(">>> ")
            except EOFError:
                break

            if not prompt.strip():
                continue

            # Handle slash commands
            if prompt.startswith("/"):
                parts = prompt[1:].split(None, 1)
                cmd_name = parts[0].lower() if parts else ""
                cmd_arg = parts[1] if len(parts) > 1 else ""

                if cmd_name in COMMANDS:
                    COMMANDS[cmd_name]['func'](state, cmd_arg)
                else:
                    print(f"  Unknown command: /{cmd_name}")
                    print(f"  Type /help for available commands.\n")
                continue

            # Generate response based on mode
            if state.mode == 'chat':
                state.chat_messages.append({'role': 'user', 'content': prompt})
                response = generate_chat(
                    state.model, state.tokenizer, state.device,
                    state.chat_messages, state.max_tokens, state.temperature, state.top_k,
                    system=state.system, block_size=state.config.block_size,
                )
                state.chat_messages.append({'role': 'assistant', 'content': response})
            elif state.mode == 'raw':
                response = generate_text(
                    state.model, state.tokenizer, state.device,
                    prompt, state.max_tokens, state.temperature, state.top_k,
                    raw=True, system=state.system,
                )
            else:  # instruction
                response = generate_text(
                    state.model, state.tokenizer, state.device,
                    prompt, state.max_tokens, state.temperature, state.top_k,
                    raw=False, system=state.system,
                )

            state.history.append((prompt, response))
            print()

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            readline.write_history_file(history_file)
        except OSError:
            pass
        print("\nBye!")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TinyGPT Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--config", type=str, default="default", choices=["default", "v2.1"],
                        help="Model config: 'default' (95M) or 'v2.1' (183M)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--mode", type=str, default=None, choices=["chat", "instruction", "raw"],
                        help="Mode: chat (default), instruction, or raw")
    parser.add_argument("--raw", action="store_true", help="Shorthand for --mode raw")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling (default: 40)")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer, config, info = load_model(args.checkpoint, device, args.config)

    if args.prompt:
        generate_text(model, tokenizer, device, args.prompt, args.max_tokens,
                      args.temperature, args.top_k, raw=args.raw, system=args.system or "")
    else:
        state = SessionState(args, model, tokenizer, config, info, device)
        interactive_loop(state)


if __name__ == "__main__":
    main()
