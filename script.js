function sendMessage() {
    let userMessage = document.getElementById("user-input").value;
    let chatBox = document.getElementById("chat-box");

    if (userMessage.trim() === "") return;

    // Display user message
    chatBox.innerHTML += `<p><strong>You:</strong> ${userMessage}</p>`;
    document.getElementById("user-input").value = "";

    // Send message to Flask backend
    fetch("/get", {
        method: "POST",
        body: JSON.stringify({ message: userMessage }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}
