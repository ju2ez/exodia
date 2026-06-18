/* Exodia ideas page: thumbs up/down voting, per-idea feedback, live best list.
 *
 * Votes are attributed to a per-browser token (so a visitor can change/retract a
 * vote, but cannot stuff the ballot from one browser). Counts come from the
 * server; the page updates optimistically and re-syncs from the API response. */
(function () {
  "use strict";

  const VOTER_KEY = "exodia_voter";
  function voterToken() {
    let t = localStorage.getItem(VOTER_KEY);
    if (!t) {
      t = "v-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem(VOTER_KEY, t);
    }
    return t;
  }
  const voter = voterToken();

  async function api(path, opts) {
    const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
    if (!res.ok) throw new Error("request failed: " + res.status);
    return res.status === 204 ? null : res.json();
  }

  // --- Voting --------------------------------------------------------------
  const myVotes = {}; // idea_id -> current value (-1/1)

  function paintVoteState(block) {
    const id = block.dataset.idea;
    const v = myVotes[id] || 0;
    block.querySelector(".vote-btn.up").classList.toggle("active", v === 1);
    block.querySelector(".vote-btn.down").classList.toggle("active", v === -1);
  }

  function applyTally(block, tally) {
    block.querySelector(".up-count").textContent = tally.up;
    block.querySelector(".down-count").textContent = tally.down;
    block.querySelector(".score").textContent = tally.score;
  }

  async function onVote(block, btn) {
    const id = block.dataset.idea;
    const clicked = parseInt(btn.dataset.value, 10);
    const next = (myVotes[id] === clicked) ? 0 : clicked; // click active button = retract
    try {
      const tally = await api(`/api/ideas/${encodeURIComponent(id)}/vote`, {
        method: "POST",
        body: JSON.stringify({ voter: voter, value: next }),
      });
      myVotes[id] = next;
      applyTally(block, tally);
      paintVoteState(block);
      refreshBest();
    } catch (e) {
      console.error(e);
    }
  }

  // --- Best ideas (re-sorted live after each vote) -------------------------
  const bestList = document.getElementById("best-list");
  async function refreshBest() {
    if (!bestList) return;
    try {
      const items = await api("/api/best?limit=10");
      bestList.innerHTML = "";
      for (const b of items) {
        const li = document.createElement("li");
        li.dataset.idea = b.idea_id;
        const a = document.createElement("a");
        a.href = "#idea-" + b.idea_id;
        a.textContent = b.title;
        li.appendChild(a);
        if (b.realized) {
          const s = document.createElement("span");
          s.className = "star"; s.title = "Realized by a published paper"; s.textContent = "★";
          li.appendChild(document.createTextNode(" "));
          li.appendChild(s);
        }
        const sc = document.createElement("span");
        sc.className = "best-score"; sc.textContent = b.score;
        li.appendChild(sc);
        bestList.appendChild(li);
      }
    } catch (e) {
      console.error(e);
    }
  }

  // --- Feedback ------------------------------------------------------------
  async function onFeedback(article, form) {
    const id = article.dataset.idea;
    const text = form.text.value.trim();
    if (!text) return;
    const author = form.author.value.trim();
    try {
      const fb = await api(`/api/ideas/${encodeURIComponent(id)}/feedback`, {
        method: "POST",
        body: JSON.stringify({ text: text, author: author || null }),
      });
      const list = article.querySelector(".feedback-list");
      const li = document.createElement("li");
      li.innerHTML = "<strong></strong> <span class='muted'></span><br>";
      li.querySelector("strong").textContent = fb.author || "anonymous";
      li.querySelector(".muted").textContent = (fb.created_utc || "").slice(0, 10);
      li.appendChild(document.createTextNode(fb.text));
      list.appendChild(li);
      const count = article.querySelector(".fb-count");
      if (count) count.textContent = list.querySelectorAll("li").length;
      form.reset();
    } catch (e) {
      console.error(e);
    }
  }

  // --- Wire up -------------------------------------------------------------
  document.querySelectorAll(".idea-vote").forEach((block) => {
    paintVoteState(block);
    block.querySelectorAll(".vote-btn").forEach((btn) =>
      btn.addEventListener("click", () => onVote(block, btn)));
  });
  document.querySelectorAll(".idea").forEach((article) => {
    const form = article.querySelector(".feedback-form");
    if (form) form.addEventListener("submit", (ev) => { ev.preventDefault(); onFeedback(article, form); });
  });

  // Highlight this browser's existing votes on load.
  api("/api/me?voter=" + encodeURIComponent(voter)).then((votes) => {
    Object.assign(myVotes, votes || {});
    document.querySelectorAll(".idea-vote").forEach(paintVoteState);
  }).catch(() => {});
})();
