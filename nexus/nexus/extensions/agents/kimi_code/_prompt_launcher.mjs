// Read the task after exec so Linux argument limits do not constrain prompt mode.
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [entry, promptFile, ...args] = process.argv.slice(2);
if (!entry || !promptFile) {
    throw new Error('Kimi Code launcher requires a CLI entry and a prompt file');
}

const cliEntry = resolve(entry);
const prompt = await readFile(promptFile, 'utf8');
process.argv = [process.execPath, cliEntry, ...args, '--prompt', prompt];
await import(pathToFileURL(cliEntry).href);
