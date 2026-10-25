import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

// Builds a fake fetch Response with the given status and JSON body.
function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const fetchMock = vi.fn<typeof fetch>();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

async function ask(text: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Your question"), `${text}{Enter}`);
  return user;
}

describe("App", () => {
  it("renders the labelled input, example chips and new conversation button", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "Weather Agent" })).toBeInTheDocument();
    expect(screen.getByLabelText("Your question")).toHaveAttribute("maxLength", "2000");
    expect(screen.getByRole("button", { name: /weather in Tokyo/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /PM2.5/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "New conversation" })).toBeInTheDocument();
  });

  it("shows the reply and sends the returned thread_id with the next message", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, { response: "Sunny, 22°C.", thread_id: "abc-123" }))
      .mockResolvedValueOnce(jsonResponse(200, { response: "Still sunny.", thread_id: "abc-123" }));
    render(<App />);

    await ask("Weather in Tokyo?");
    expect(await screen.findByText("Sunny, 22°C.")).toBeInTheDocument();

    await ask("And tomorrow?");
    await screen.findByText("Still sunny.");

    const firstBody = JSON.parse(fetchMock.mock.calls[0][1]!.body as string);
    const secondBody = JSON.parse(fetchMock.mock.calls[1][1]!.body as string);
    expect(firstBody).toEqual({ message: "Weather in Tokyo?", thread_id: null });
    expect(secondBody).toEqual({ message: "And tomorrow?", thread_id: "abc-123" });
  });

  it("shows the friendly guardrail message on 400 and never the raw detail", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(400, { detail: "Message flagged as a potential prompt injection attempt." }),
    );
    render(<App />);

    await ask("some sneaky message");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "That message was blocked by the safety checks. Please rephrase your question.",
    );
    expect(screen.queryByText(/prompt injection/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    // the text is handed back so it can be rephrased
    expect(screen.getByLabelText("Your question")).toHaveValue("some sneaky message");
  });

  it("shows the error state with Retry on 500, and Retry resends the message", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(500, { detail: "Agent error: Traceback ..." }))
      .mockResolvedValueOnce(jsonResponse(200, { response: "Sunny, 22°C.", thread_id: "abc-123" }));
    render(<App />);

    const user = await ask("Weather in Tokyo?");

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong. Please try again.");
    expect(screen.queryByText(/Traceback/)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Sunny, 22°C.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getAllByText("Weather in Tokyo?")).toHaveLength(1); // not duplicated
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("treats a network failure like a server error", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    render(<App />);

    await ask("Weather in Tokyo?");

    expect(await screen.findByRole("alert")).toHaveTextContent("Something went wrong. Please try again.");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("New conversation clears the messages and the thread_id", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, { response: "Sunny, 22°C.", thread_id: "abc-123" }))
      .mockResolvedValueOnce(jsonResponse(200, { response: "Hello again.", thread_id: "new-456" }));
    render(<App />);

    const user = await ask("Weather in Tokyo?");
    await screen.findByText("Sunny, 22°C.");

    await user.click(screen.getByRole("button", { name: "New conversation" }));
    expect(screen.queryByText("Sunny, 22°C.")).not.toBeInTheDocument();

    await ask("Hi");
    await screen.findByText("Hello again.");
    const body = JSON.parse(fetchMock.mock.calls[1][1]!.body as string);
    expect(body.thread_id).toBeNull();
  });
});
