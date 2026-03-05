// Backend API base URL exposed by docker-compose.
const API_BASE_URL = "http://localhost:8000";

// Grab key DOM nodes once so we can reuse them in event handlers.
// - `chat` is the container where all messages are rendered.
// - `form` is the question form that triggers a backend call.
// - `input` is the text field where the user types the question.
const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const input = document.getElementById("question");

// Renders one message bubble in the chat window.
function appendMessage(role, text, meta = "") {
  // Create one wrapper element for the message and assign a role-based class
  // so CSS can style user and assistant messages differently.
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.textContent = text;

  // Optional metadata block shown under a message.
  // This is used for things like sources or debug details.
  if (meta) {
    const metaLine = document.createElement("div");
    metaLine.className = "meta";
    metaLine.textContent = meta;
    wrapper.appendChild(metaLine);
  }

  // Insert the message into the chat timeline.
  chat.appendChild(wrapper);
  // Always keep the latest message visible.
  chat.scrollTop = chat.scrollHeight;
}

// Sends a user query to the backend /chat endpoint.
async function askQuestion(query) {
  // POST the user question to backend and wait for JSON response.
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query }),
  });

  // Convert non-2xx HTTP status into a thrown error so the caller can show
  // a user-friendly error message in one centralized place.
  if (!response.ok) {
    const err = await response.text();
    throw new Error(err || "Request failed");
  }

  // Return parsed JSON (shape expected: answer, sources, augmented_query).
  return response.json();
}

// Main submit workflow for the chat form.
// Sequence:
// 1) Prevent full page reload.
// 2) Validate query is not empty.
// 3) Render user message immediately.
// 4) Render temporary assistant placeholder.
// 5) Call backend and replace placeholder with final answer + metadata.
form.addEventListener("submit", async (event) => {
  // Keep SPA behavior: do not let browser submit and navigate.
  event.preventDefault();

  // Trim whitespace so inputs like "   " are treated as empty.
  const query = input.value.trim();

  // Guard clause: stop early if query is empty.
  if (!query) {
    return;
  }

  // Show user message in chat immediately for responsive UX.
  appendMessage("user", query);

  // Reset input and move cursor back to input for the next question.
  input.value = "";
  input.focus();

  // Show a temporary placeholder while waiting for backend response.
  appendMessage("assistant", "Thinking...");

  // Keep reference to the placeholder node so we can update it in place
  // instead of creating a second assistant bubble.
  const thinkingNode = chat.lastChild;

  try {
    // Wait for backend answer.
    const result = await askQuestion(query);

    // Build source summary text when sources are available.
    // If no sources are returned, show an explicit fallback message.
    const sources = result.sources?.length
      ? `Sources: ${result.sources.join(", ")}`
      : "No sources found";

    // Replace the placeholder text with the actual model answer.
    thinkingNode.textContent = result.answer || "No answer returned.";

    // Add metadata line under the answer with source files and rewritten query.
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${sources} | Augmented query: ${result.augmented_query}`;
    thinkingNode.appendChild(meta);
  } catch (error) {
    // If request/network/backend fails, replace placeholder with error text
    // so user sees failure in chat timeline.
    thinkingNode.textContent = `Error: ${error.message}`;
  }
});
