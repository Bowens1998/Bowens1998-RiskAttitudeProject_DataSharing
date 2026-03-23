// subject-id.js — Shared Subject ID manager
// Include this script BEFORE any experiment driver script on every page.
//
// Lifecycle:
//   1. If 'all_submitted' is 'true', a previous session is fully done.
//      → Clear all session data and generate a fresh Subject ID.
//   2. If no 'subject_id' exists, generate one (first visit).
//   3. Expose window.SUBJECT_ID for use by experiment drivers.

(function () {
    'use strict';

    // Keys managed by this module
    const SESSION_KEYS = [
        'subject_id',
        'all_submitted',
        'task_complete_dsb',
        'task_complete_fip',
        'task_complete_tpb',
        'experiment_data_dsb',
        'experiment_data_fip',
        'experiment_data_tpb',
        'dsb_backup_log'
    ];

    // Generate a UUID v4
    function generateUUID() {
        // Use crypto.randomUUID if available (modern browsers)
        if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
            return crypto.randomUUID();
        }
        // Fallback for older browsers
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            var r = Math.random() * 16 | 0;
            var v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // If previous session was fully submitted, wipe everything and start fresh
    if (localStorage.getItem('all_submitted') === 'true') {
        SESSION_KEYS.forEach(function (key) {
            localStorage.removeItem(key);
        });
    }

    // Ensure a Subject ID exists
    if (!localStorage.getItem('subject_id')) {
        localStorage.setItem('subject_id', generateUUID());
    }

    // Expose globally
    window.SUBJECT_ID = localStorage.getItem('subject_id');
})();
