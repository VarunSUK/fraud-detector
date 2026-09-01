package models

import "testing"

func TestValidate(t *testing.T) {
	tests := []struct {
		name    string
		txn     Transaction
		wantErr bool
	}{
		{"valid transaction", Transaction{Amount: 100, Time: 1000}, false},
		{"zero amount and time", Transaction{Amount: 0, Time: 0}, false},
		{"negative amount", Transaction{Amount: -1, Time: 1000}, true},
		{"negative time", Transaction{Amount: 100, Time: -1}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.txn.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestGetFeatureVector(t *testing.T) {
	txn := Transaction{Time: 1000, Amount: 250.5, V1: 1.1, V28: 2.2}
	vec := txn.GetFeatureVector()

	// Time + V1..V28 + Amount
	wantLen := 30
	if len(vec) != wantLen {
		t.Fatalf("GetFeatureVector() length = %d, want %d", len(vec), wantLen)
	}
	if vec[0] != 1000 {
		t.Errorf("vec[0] (time) = %v, want 1000", vec[0])
	}
	if vec[1] != 1.1 {
		t.Errorf("vec[1] (v1) = %v, want 1.1", vec[1])
	}
	if vec[len(vec)-1] != 250.5 {
		t.Errorf("last element (amount) = %v, want 250.5", vec[len(vec)-1])
	}
}

func TestValidationError_Error(t *testing.T) {
	err := &ValidationError{Field: "amount", Message: "amount must be non-negative"}
	want := "amount: amount must be non-negative"
	if err.Error() != want {
		t.Errorf("Error() = %q, want %q", err.Error(), want)
	}
}
