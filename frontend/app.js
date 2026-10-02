const form = document.getElementById("research-form");

const questionInput = document.getElementById("question");

const button = document.getElementById("research-button");

const errorBox = document.getElementById("error");

const results = document.getElementById("results");

const resultQuestion =
    document.getElementById("result-question");

const planList =
    document.getElementById("plan");

const findingsContainer =
    document.getElementById("findings");

const report =
    document.getElementById("report");


form.addEventListener("submit", async (event) => {

    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }


    // Reset UI

    errorBox.classList.add("hidden");

    results.classList.add("hidden");

    button.disabled = true;

    button.textContent = "Researching...";


    try {

        const response = await fetch("http://127.0.0.1:8000/research", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Research request failed."
            );

        }


        displayResults(data);


    } catch (error) {

        errorBox.textContent = error.message;

        errorBox.classList.remove("hidden");

    } finally {

        button.disabled = false;

        button.textContent = "Start Research →";

    }

});


function displayResults(data) {

    results.classList.remove("hidden");


    // Question

    resultQuestion.textContent =
        data.question;


    // Plan

    planList.innerHTML = "";

    for (const subquestion of data.plan.subquestions) {

        const li = document.createElement("li");

        li.textContent = subquestion;

        planList.appendChild(li);

    }


    // Findings

    findingsContainer.innerHTML = "";


    data.findings.forEach((item, index) => {

    const finding = document.createElement("article");

    const failed = item.finding === null;

    finding.className = failed
        ? "finding finding-error"
        : "finding";

    finding.innerHTML = `
        <span class="finding-number">
            ${String(index + 1).padStart(2, "0")}
        </span>

        <div>
            <h5></h5>
            <p></p>
        </div>
    `;

    finding.querySelector("h5").textContent =
        failed
            ? `⚠️ ${item.subquestion}`
            : `✓ ${item.subquestion}`;

    finding.querySelector("p").textContent =
        failed
            ? item.error || "This subquestion failed."
            : item.finding;

    findingsContainer.appendChild(finding);
});


    // Final report

    const reportFailed = Boolean(
        data.report === null && data.report_error
    );

    report.textContent =
        reportFailed ? data.report_error : data.report;

    report.classList.toggle("error", reportFailed);


    // Scroll to results

    results.scrollIntoView({
        behavior: "smooth"
    });

}
