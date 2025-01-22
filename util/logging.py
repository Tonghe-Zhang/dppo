import shutil
def create_bordered_table(left_column, right_column, border_char='#'):
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Calculate column widths
    left_width = terminal_width // 2 - 4  # -4 for '# ' and ' #' on sides
    right_width = terminal_width - left_width - 4  # -4 for '# ' and ' #' on sides

    # Split the text into lines and center/align each line
    left_lines = left_column.split('\n')
    right_lines = right_column.split('\n')
    formatted_lines = [
        f"# {left_line.ljust(left_width)} {right_line.rjust(right_width)} #"
        for left_line, right_line in zip(left_lines, right_lines)
    ]

    # Create top and bottom borders
    border = border_char * terminal_width

    # Construct the final message
    message = (
        f"{border}\n"
        + '\n'.join(formatted_lines)
        + f"\n{border}"
    )

    return message