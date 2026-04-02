const textInput = document.getElementById("textInput");
const fileInput = document.getElementById("fileInput");
const analyzeTextBtn = document.getElementById("analyzeTextBtn");
const analyzeFileBtn = document.getElementById("analyzeFileBtn");
const resultSection = document.getElementById("resultSection");
const statusEl = document.getElementById("status");

const summaryGrid = document.getElementById("summaryGrid");
const misspellingsBody = document.getElementById("misspellingsBody");
const grammarList = document.getElementById("grammarList");
const correctedText = document.getElementById("correctedText");
const tokenBody = document.getElementById("tokenBody");

const SUMMARY_LABELS = {
  character_count: "Characters",
  token_count: "Tokens",
  word_count: "Words",
  misspelled_word_count: "Misspelled Words",
  unique_misspelled_word_count: "Unique Misspellings",
  grammar_issue_count: "Grammar Issues",
};

function updateStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#b42318" : "#607060";
}

function renderSummary(summary) {
  summaryGrid.innerHTML = "";
  Object.entries(summary).forEach(([key, value]) => {
    const card = document.createElement("div");
    card.className = "summary-card";
    card.innerHTML = `
      <p class="label">${SUMMARY_LABELS[key] || key}</p>
      <p class="value">${value}</p>
    `;
    summaryGrid.appendChild(card);
  });
}

function renderMisspellings(misspellings) {
  misspellingsBody.innerHTML = "";
  if (!misspellings.length) {
    misspellingsBody.innerHTML = '<tr><td colspan="3">No misspellings detected.</td></tr>';
    return;
  }

  misspellings.forEach((entry) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${entry.word}</td>
      <td>${entry.count}</td>
      <td>${entry.suggestions?.join(", ") || "No suggestion"}</td>
    `;
    misspellingsBody.appendChild(row);
  });
}

function renderGrammar(grammarIssues) {
  grammarList.innerHTML = "";
  if (!grammarIssues.length) {
    const li = document.createElement("li");
    li.textContent = "No grammar issues detected by current rules.";
    grammarList.appendChild(li);
    return;
  }

  grammarIssues.forEach((issue) => {
    const li = document.createElement("li");
    const cls = issue.severity === "warning" ? "sev-warning" : "sev-info";
    li.className = cls;
    li.textContent = `[${issue.rule}] ${issue.message}`;
    grammarList.appendChild(li);
  });
}

function renderTokens(tokens) {
  tokenBody.innerHTML = "";
  tokens.slice(0, 120).forEach((token) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${token.lexeme.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</td>
      <td>${token.token_type}</td>
      <td>${token.line}</td>
      <td>${token.column}</td>
    `;
    tokenBody.appendChild(row);
  });
}

function renderResult(result) {
  renderSummary(result.summary);
  renderMisspellings(result.misspellings);
  renderGrammar(result.grammar_issues);
  renderTokens(result.tokens);
  correctedText.textContent = result.corrected_text;
  resultSection.classList.remove("hidden");
}

async function handleResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Analysis failed.");
  }
  return data;
}

analyzeTextBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    updateStatus("Please enter text before analyzing.", true);
    return;
  }

  try {
    updateStatus("Running lexical + spelling + grammar analysis...");
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const result = await handleResponse(response);
    renderResult(result);
    updateStatus("Analysis complete.");
  } catch (error) {
    updateStatus(error.message, true);
  }
});

analyzeFileBtn.addEventListener("click", async () => {
  const selected = fileInput.files?.[0];
  if (!selected) {
    updateStatus("Select a .txt or .md file first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", selected);

  try {
    updateStatus("Uploading and analyzing file...");
    const response = await fetch("/api/analyze-file", {
      method: "POST",
      body: formData,
    });

    const result = await handleResponse(response);
    renderResult(result);
    updateStatus(`Analysis complete for file: ${result.source_file || selected.name}.`);
  } catch (error) {
    updateStatus(error.message, true);
  }
});
