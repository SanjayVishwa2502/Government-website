// Show or hide the chatbot
document.getElementById('chatbot-btn').addEventListener('click', () => {
    document.getElementById('chatbot-container').style.display = 'block';
});

document.getElementById('close-btn').addEventListener('click', () => {
    document.getElementById('chatbot-container').style.display = 'none';
});

// Handle sending messages
document.getElementById('send-btn').addEventListener('click', () => {
    const userMessage = document.getElementById('user-input').value.trim();
    const chatArea = document.getElementById('chat-area');

    if (userMessage) {
        // Display user's message
        const userDiv = document.createElement('div');
        userDiv.classList.add('user-message', 'message');
        userDiv.textContent = userMessage;
        chatArea.appendChild(userDiv);

        // Clear input field
        document.getElementById('user-input').value = '';

        // Simulate chatbot response
        setTimeout(() => {
            const botResponse = document.createElement('div');
            botResponse.classList.add('bot-message', 'message');
            botResponse.textContent = "Thanks for your message. I'll get back to you shortly!";
            chatArea.appendChild(botResponse);

            // Scroll to bottom
            chatArea.scrollTop = chatArea.scrollHeight;
        }, 1000);
    }
});
