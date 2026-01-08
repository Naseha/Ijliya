// Point to your LIVE Render backend
const IJLIYA_API_URL = "https://ijliya.onrender.com/ask";

// Add message to chat
function addMessage(text, sender) {
  const chat = document.getElementById("chat");
  const msg = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.innerHTML = text;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight; // auto-scroll
}

// Initialize chat with tribute
document.addEventListener("DOMContentLoaded", () => {
  addMessage(
    "I’m Ijiliya — named for the quiet ones who built knowledge in silence.<br><br>Ask me anything. I’ll show you only what’s in Wikipedia and Wikimedia.",
    "ijiliya"
  );
});

// Handle user question
async function askIjliya() {
  const input = document.getElementById("userQuery");
  const query = input.value.trim();
  const errorEl = document.getElementById("error");

  if (!query) return;

  // Show user message
  addMessage(query, "user");
  input.value = "";

  // Clear error
  errorEl.className = "error hidden";

  try {
    const response = await fetch(IJLIYA_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query })
    });

    if (!response.ok) throw new Error("Service unavailable");

    const data = await response.json();

    let reply;
    if (data.wiki_links && data.wiki_links.length > 0) {
      const link = data.wiki_links[0];
      const title = data.title || "Answer";
      const extract = data.extract ? data.extract.substring(0, 200) + "..." : "";
      reply = `
        <strong>${title}</strong><br>
        ${extract}<br>
        <a href="${link}" target="_blank" rel="noopener" style="color:#1a73e8; text-decoration:underline;">Read more</a><br>
        <small>Source: ${data.source}</small>
      `;
    } else {
      reply = "I couldn’t find a Wikipedia page on that. Try rephrasing?";
    }

    addMessage(reply, "ijiliya");
  } catch (err) {
    addMessage("⚠️ Ijiliya is unreachable. Please try again later.", "ijiliya");
  }
}

// Button & Enter key
document.getElementById("askBtn").addEventListener("click", askIjliya);
document.getElementById("userQuery").addEventListener("keypress", (e) => {
  if (e.key === "Enter") askIjliya();
});
