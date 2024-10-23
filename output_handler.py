import sys
import time

def colored_print(message: str, color_code: str):
    print(f"\033[{color_code}m{message}\033[0m", flush=True)

if __name__ == "__main__":
    while True:
        try:
            with open('trade_messages.txt', 'r') as f:
                lines = f.readlines()
            
            if lines:
                with open('trade_messages.txt', 'w') as f:
                    pass  # Clear the file

                for line in lines:
                    line = line.strip()
                    if line.startswith("STOP-LOSS"):
                        colored_print(line, "91")  # Red for stop-loss
                    elif line.startswith("BUY"):
                        colored_print(line, "92")  # Green for buy
                    elif line.startswith("SELL"):
                        colored_print(line, "91")  # Red for sell
            
            time.sleep(0.1)  # Short delay to prevent excessive CPU usage
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            time.sleep(1)  # Longer delay if an error occurs