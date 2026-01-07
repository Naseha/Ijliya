// Point to your LIVE Render backend
const IJLIYA_API_URL = "https://ijliya.onrender.com/ask";

async function askIjliya() {
  const input = document.getElementById("userQuery");
  const query = input.value.trim();
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");
  const errorEl = document.getElementById("error");

  if (!query) return;

  // Reset UI
  resultsEl.innerHTML = "";
  errorEl.className = "error hidden";
  statusEl.className = "status";

  try {
    const response = await fetch(IJLIYA_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query })
    });

    statusEl.className = "status hidden";

    if (!response.ok) throw new Error("Service unavailable");

    const data = await response.json();

    if (data.wiki_links && data.wiki_links.length > 0) {
      const link = data.wiki_links[0];
      const title = data.title || "Answer";
      resultsEl.innerHTML = `
        <div class="result-item">
          <div class="result-title">${title}</div>
          <a class="result-link" href="${link}" target="_blank" rel="noopener">${link}</a>
          ${data.extract ? `<div class="result-extract">${data.extract.substring(0, 200)}...</div>` : ''}
          <div class="small">Source: ${data.source}</div>
        </div>
      `;
    } else {
      resultsEl.innerHTML = `<div>No Wikipedia page found.</div>`;
    }
  } catch (err) {
    statusEl.className = "status hidden";
    errorEl.textContent = "⚠️ Ijliya is unreachable. Please try again later.";
    errorEl.className = "error";
  }
}

// Allow Enter key to submit
document.getElementById("userQuery").addEventListener("keypress", (e) => {
  if (e.key === "Enter") askIjliya();
});
