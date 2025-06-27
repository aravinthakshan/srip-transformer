#!/bin/bash

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unsupported OS"
    exit 1
fi

# Install venv package if needed
install_venv_package() {
    case "$OS" in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y python3-venv
            ;;
        arch)
            sudo pacman -Sy --noconfirm python-virtualenv
            ;;
        *)
            echo "Unsupported distro: $OS"
            exit 1
            ;;
    esac
}

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found!"
    exit 1
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    install_venv_package
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate and install requirements
source venv/bin/activate

if [ -f requirements.txt ]; then
    echo "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi
