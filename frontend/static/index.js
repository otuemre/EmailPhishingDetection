const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const loadingOverlay = document.getElementById("loading-overlay");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  resultBox.textContent = "";
  loadingOverlay.style.display = "flex";

  const formData = new FormData(form);
  const data = Object.fromEntries(formData.entries());

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const res = await response.json();
    resultBox.innerText = res.result;
  } catch (err) {
    resultBox.textContent = "Error: Could not connect to the API";
  } finally {
    loadingOverlay.style.display = "none";
  }
});
