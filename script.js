(() => {
  "use strict";

  const FEATURE_ORDER = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

  // Real column statistics from df.describe() on the full 284,807-row dataset.
  // Used only to generate plausible-looking demo values for V1-V28 (all have
  // mean ~0 by construction, since they're PCA components) -- never presented
  // as real transactions.
  const FEATURE_STATS = {
    V1: 1.9587, V2: 1.6513, V3: 1.5163, V4: 1.4159, V5: 1.3802, V6: 1.3323,
    V7: 1.2371, V8: 1.1944, V9: 1.0986, V10: 1.0888, V11: 1.0207, V12: 0.9992,
    V13: 0.9953, V14: 0.9586, V15: 0.9153, V16: 0.8763, V17: 0.8493, V18: 0.8382,
    V19: 0.8140, V20: 0.7709, V21: 0.7345, V22: 0.7257, V23: 0.6245, V24: 0.6056,
    V25: 0.5213, V26: 0.4822, V27: 0.4036, V28: 0.3301,
  };

  // Five real rows from the dataset (df.head()), each an actual legitimate
  // transaction. Real fraud cases make up only 0.17% of the data, so none
  // happen to fall in the first five rows -- that scarcity is the point of
  // the whole project, not an omission.
  const REAL_SAMPLES = [
    { Time: 0, V1: -1.3598, V2: -0.0728, V3: 2.5363, V4: 1.3782, V5: -0.3383, V6: 0.4624, V7: 0.2396, V8: 0.0987, V9: 0.3638, V10: 0.0908, V11: -0.5516, V12: -0.6178, V13: -0.9914, V14: -0.3112, V15: 1.4682, V16: -0.4704, V17: 0.2080, V18: 0.0258, V19: 0.4040, V20: 0.2514, V21: -0.0183, V22: 0.2778, V23: -0.1105, V24: 0.0669, V25: 0.1285, V26: -0.1891, V27: 0.1336, V28: -0.0211, Amount: 149.62 },
    { Time: 0, V1: 1.1919, V2: 0.2662, V3: 0.1665, V4: 0.4482, V5: 0.0600, V6: -0.0824, V7: -0.0788, V8: 0.0851, V9: -0.2554, V10: -0.1670, V11: 1.6127, V12: 1.0652, V13: 0.4891, V14: -0.1438, V15: 0.6356, V16: 0.4639, V17: -0.1148, V18: -0.1834, V19: -0.1458, V20: -0.0691, V21: -0.2258, V22: -0.6387, V23: 0.1013, V24: -0.3398, V25: 0.1672, V26: 0.1259, V27: -0.0090, V28: 0.0147, Amount: 2.69 },
    { Time: 1, V1: -1.3584, V2: -1.3402, V3: 1.7732, V4: 0.3798, V5: -0.5032, V6: 1.8005, V7: 0.7915, V8: 0.2477, V9: -1.5147, V10: 0.2076, V11: 0.6245, V12: 0.0661, V13: 0.7173, V14: -0.1659, V15: 2.3459, V16: -2.8901, V17: 1.1100, V18: -0.1214, V19: -2.2619, V20: 0.5250, V21: 0.2480, V22: 0.7717, V23: 0.9094, V24: -0.6893, V25: -0.3276, V26: -0.1391, V27: -0.0554, V28: -0.0598, Amount: 378.66 },
    { Time: 1, V1: -0.9663, V2: -0.1852, V3: 1.7930, V4: -0.8633, V5: -0.0103, V6: 1.2472, V7: 0.2376, V8: 0.3774, V9: -1.3870, V10: -0.0550, V11: -0.2265, V12: 0.1782, V13: 0.5078, V14: -0.2879, V15: -0.6314, V16: -1.0596, V17: -0.6841, V18: 1.9658, V19: -1.2326, V20: -0.2080, V21: -0.1083, V22: 0.0053, V23: -0.1903, V24: -1.1756, V25: 0.6474, V26: -0.2219, V27: 0.0627, V28: 0.0615, Amount: 123.50 },
    { Time: 2, V1: -1.1582, V2: 0.8777, V3: 1.5487, V4: 0.4030, V5: -0.4072, V6: 0.0959, V7: 0.5929, V8: -0.2705, V9: 0.8177, V10: 0.7531, V11: -0.8228, V12: 0.5382, V13: 1.3459, V14: -1.1197, V15: 0.1751, V16: -0.4514, V17: -0.2370, V18: -0.0382, V19: 0.8035, V20: 0.4085, V21: -0.0094, V22: 0.7983, V23: -0.1375, V24: 0.1413, V25: -0.2060, V26: 0.5023, V27: 0.2194, V28: 0.2152, Amount: 69.99 },
  ];

  const form = document.getElementById("txnForm");
  const vGrid = document.getElementById("vGrid");
  const amountInput = document.getElementById("amount");
  const timeInput = document.getElementById("time");
  const resultEl = document.getElementById("result");
  const formNote = document.getElementById("formNote");
  const submitBtn = document.getElementById("submitBtn");

  let sampleIndex = -1;

  function buildGrid() {
    const frag = document.createDocumentFragment();
    Object.keys(FEATURE_STATS).forEach((name) => {
      const wrap = document.createElement("div");
      wrap.className = "v-field";

      const label = document.createElement("label");
      label.setAttribute("for", `f-${name}`);
      label.textContent = name;

      const input = document.createElement("input");
      input.type = "number";
      input.step = "any";
      input.id = `f-${name}`;
      input.name = name;
      input.value = "0";
      input.required = true;

      wrap.appendChild(label);
      wrap.appendChild(input);
      frag.appendChild(wrap);
    });
    vGrid.appendChild(frag);
  }

  function setVValues(values) {
    Object.keys(FEATURE_STATS).forEach((name) => {
      const el = document.getElementById(`f-${name}`);
      if (el && values[name] !== undefined) el.value = values[name];
    });
  }

  // Box-Muller transform for an approximately normal random value.
  function gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function loadSample() {
    sampleIndex = (sampleIndex + 1) % REAL_SAMPLES.length;
    const sample = REAL_SAMPLES[sampleIndex];
    amountInput.value = sample.Amount;
    timeInput.value = sample.Time;
    setVValues(sample);
    formNote.textContent = `Loaded real transaction ${sampleIndex + 1} of ${REAL_SAMPLES.length} from the dataset — legitimate, like about 99.8% of it.`;
    clearResult();
  }

  function randomize() {
    const values = {};
    Object.entries(FEATURE_STATS).forEach(([name, std]) => {
      values[name] = (gaussianRandom() * std).toFixed(4);
    });
    setVValues(values);
    formNote.textContent = "V1–V28 randomized from the dataset's real per-feature spread — a simulated input, not an actual transaction.";
    clearResult();
  }

  function clearResult() {
    resultEl.className = "result result--empty";
    resultEl.innerHTML = `<p class="result__placeholder">Submit the transaction above to see a prediction.</p>`;
  }

  function renderResult(data) {
    const isFraud = data.prediction === 1;
    resultEl.className = `result result--${isFraud ? "fraud" : "legit"}`;
    const pct = (data.fraud_probability * 100).toFixed(2);
    resultEl.innerHTML = `
      <p class="result__verdict">${data.label}</p>
      <p class="result__probability">Estimated fraud probability: <strong>${pct}%</strong></p>
    `;
  }

  function renderError(message) {
    resultEl.className = "result result--error";
    resultEl.innerHTML = `<p class="result__verdict">Couldn't score that transaction</p><p class="result__probability">${message}</p>`;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const payload = {};
    for (const name of FEATURE_ORDER) {
      const el = document.getElementById(name === "Amount" ? "amount" : name === "Time" ? "time" : `f-${name}`);
      const num = Number(el.value);
      if (el.value === "" || Number.isNaN(num)) {
        formNote.textContent = `${name} needs a number.`;
        el.focus();
        return;
      }
      payload[name] = num;
    }

    formNote.textContent = "";
    submitBtn.disabled = true;
    submitBtn.textContent = "Checking…";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server responded with ${res.status}`);
      }
      const data = await res.json();
      renderResult(data);
    } catch (err) {
      renderError(err.message || "The request failed. Is the API running?");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Check transaction";
    }
  }

  buildGrid();
  document.getElementById("loadSampleBtn").addEventListener("click", loadSample);
  document.getElementById("randomizeBtn").addEventListener("click", randomize);
  form.addEventListener("submit", handleSubmit);
})();
