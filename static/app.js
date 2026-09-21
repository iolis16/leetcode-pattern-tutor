const input = document.getElementById("problem-input");
const kSelect = document.getElementById("k-select");
const btn = document.getElementById("analyze-btn");
const resultSection = document.getElementById("result-section");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result-box");
const patternsEl = document.getElementById("patterns");
const confidenceEl = document.getElementById("confidence");
const reasoningEl = document.getElementById("reasoning");
const retrievedListEl = document.getElementById("retrieved-list");

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderResult(data) {
  patternsEl.innerHTML = data.patterns
    .map((p) => `<span class="pattern-chip">${escapeHtml(p)}</span>`)
    .join("");
  confidenceEl.textContent = `${data.confidence} confidence`;
  reasoningEl.textContent = data.reasoning;

  const citedTitles = new Set(data.cited_problem_titles);
  retrievedListEl.innerHTML = data.retrieved
    .map((r) => {
      const isCited = citedTitles.has(r.title);
      const tags = r.tags
        .map((t) => `<span class="tag-chip">${escapeHtml(t)}</span>`)
        .join("");
      const leetcodeUrl = `https://leetcode.com/problems/${encodeURIComponent(r.slug)}/`;
      return `
        <li class="retrieved-item${isCited ? " cited" : ""}">
          <div class="retrieved-title">
            <a href="${leetcodeUrl}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title)}</a>${isCited ? " (cited)" : ""}
          </div>
          <div class="retrieved-meta">${escapeHtml(r.difficulty)} &middot; similarity ${r.similarity.toFixed(3)}</div>
          <div class="retrieved-tags">${tags}</div>
        </li>`;
    })
    .join("");
}

async function analyze() {
  const problemStatement = input.value.trim();
  if (problemStatement.length < 20) {
    errorBox.textContent = "Please paste a fuller problem statement (at least 20 characters).";
    errorBox.hidden = false;
    resultBox.hidden = true;
    resultSection.hidden = false;
    return;
  }

  btn.disabled = true;
  btn.textContent = "Analyzing...";
  errorBox.hidden = true;
  resultBox.hidden = true;
  resultSection.hidden = false;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        problem_statement: problemStatement,
        k: Number(kSelect.value),
      }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResult(data);
    resultBox.hidden = false;
  } catch (err) {
    errorBox.textContent = err.message || "Something went wrong.";
    errorBox.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

btn.addEventListener("click", analyze);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    analyze();
  }
});
