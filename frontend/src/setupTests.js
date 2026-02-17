// Polyfill TextEncoder/TextDecoder for JSDOM (needed by react-router v7)
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// jest-dom adds custom jest matchers for asserting on DOM nodes.
import '@testing-library/jest-dom';

// Mock ResizeObserver (required by Recharts ResponsiveContainer in JSDOM)
global.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};
